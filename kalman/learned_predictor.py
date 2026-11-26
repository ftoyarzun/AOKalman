"""VARN-LQG -- the steady-state Kalman predictor learnt end to end on the telemetry (torch, CPU).

varn_lqg's predictor, with its matrices trained to minimise the residual it would leave, instead
of being derived from a model (A by least squares, Q, R, Riccati):

    x'_k = A x_{k-1},  A = [[var], [I 0]]          VAR(N) companion form
    x_k  = x'_k + K (pol_k - H x'_k)               constant gain
    p_k  = C x_k                                    the turbulence, delay frames ahead
    res_t = pol_t - (1 - f) p_{t-n} - f p_{t-n-1}   delay = n + f; the controller outputs -p

    loss = mean ||res_t||^2 over the telemetry

res_t is the pseudo-open-loop residual: what is left of pol_t after the commands computed n
(and n + 1) frames before. pol holds the WFS noise, but the noise is white: it cannot be
predicted, so it does not bias the minimiser (least squares on pol is biased by it).

Training: the predictor is linear and time invariant, so p_k = sum_j G_j pol_{k-j} with
G_j = C F^j K, F = (I - K H) A, truncated to `taps` frames. Each Adam iteration scores a random
batch of frames through it (one product, no recursion). Early stopping, and the choice of the
result, use the exact recursive filter on the last part of the telemetry, held out.

The trained (var, K, C) have the shapes of varn_lqg's params "var", "kalman", "predict".
"""
import time

import numpy as np
import torch


def split_delay(delay):
    """delay = n + f frames, as rama_cup: n whole frames and the fraction f."""
    delay = max(1.0, float(delay))
    n = int(delay + 1e-9)
    f = delay - n
    return n, (f if f > 1e-9 else 0.0)


def filter_matrix(var, K):
    """F = (I - K H) A = A - K var, the state's own dynamics in the filter: it must be stable."""
    n, size = var.shape
    A = np.zeros((size, size))
    A[:n] = var
    A[n:, :-n] = np.eye(size - n)
    return A - K @ var


def spectral_radius(var, K):
    return float(np.max(np.abs(np.linalg.eigvals(filter_matrix(np.asarray(var, float), np.asarray(K, float))))))


def hidden_loop_radius(var, K, C, delay, rho):
    """
    Stability of the loop the controller closes through its own pseudo open loop, when the WFS
    sees rho times the residual the controller assumes (rho = true og / og used; scalar or per
    mode). The controller rebuilds pol = z / og + applied, so

        pol_k = rho phi_k + (1 - rho) applied_k,   applied_k = (1 - f) C x_{k-n} + f C x_{k-n-1}
        x_k   = F x_{k-1} + K pol_k

    and the controller's own output comes back with gain (1 - rho): a loop x_k = F x_{k-1} +
    K (1 - rho) applied_k. Returns its spectral radius: < 1 stable. rho = 1 gives that of F.
    """
    var, K, C = (np.asarray(m, float) for m in (var, K, C))
    n_delay, f_delay = split_delay(delay)
    F = filter_matrix(var, K)
    size = F.shape[0]
    leak = K @ (np.diag(1.0 - np.broadcast_to(rho, (C.shape[0],))) @ C)
    blocks = n_delay + (1 if f_delay else 0)                 # state [x_{k-1}, ..., x_{k-blocks}]
    M = np.zeros((blocks * size, blocks * size))
    M[:size, :size] = F
    M[:size, (n_delay - 1) * size:n_delay * size] += (1.0 - f_delay) * leak
    if f_delay:
        M[:size, n_delay * size:] += f_delay * leak
    M[size:, :-size] = np.eye((blocks - 1) * size)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def robust_mix(start, learnt, delay, rhos=(0.7, 1.0, 1.3), max_radius=1.0, steps=6):
    """
    The largest step alpha from `start` towards `learnt`, (1 - alpha) start + alpha learnt for
    each matrix, whose hidden loop (hidden_loop_radius) stays below max_radius for every rho in
    rhos: errors on the WFS's optical gain, from x 0.7 to x 1.3 by default. Learning only for the
    residual pushes the predictor towards the edge of stability, with no margin for an optical gain
    that is not exactly what the controller assumes (a pyramid's changes with the residual).

    Returns (var, K, C), alpha, and the worst radius over rhos.
    """
    start = [np.asarray(m, float) for m in start]
    learnt = [np.asarray(m, float) for m in learnt]

    def mix(alpha):
        return [(1.0 - alpha) * a + alpha * b for a, b in zip(start, learnt)]

    def worst(alpha):
        return max(hidden_loop_radius(*mix(alpha), delay, r) for r in rhos)

    w1 = worst(1.0)
    if w1 <= max_radius:
        return tuple(learnt), 1.0, w1
    w0 = worst(0.0)
    if w0 > max_radius:                                      # the start itself has no margin: keep it
        return tuple(start), 0.0, w0
    lo, hi, w_lo = 0.0, 1.0, w0
    for _ in range(steps):                                   # bisection on alpha
        mid = 0.5 * (lo + hi)
        w = worst(mid)
        if w <= max_radius:
            lo, w_lo = mid, w
        else:
            hi = mid
    return tuple(mix(lo)), lo, w_lo


def _impulse_response(var, K, C, taps):
    """G [taps, n, n], G_j = C F^j K: the predictor as a filter on the past pol (differentiable)."""
    n, size = var.shape
    A = torch.cat([var, torch.eye(size - n, size, dtype=var.dtype)], dim=0) if size > n else var
    F = A - K @ var
    M, G = K, []
    for _ in range(taps):
        G.append(C @ M)
        M = F @ M
    return torch.stack(G)


def _residuals(seq, var, K, C, n_delay, f_delay, warm, target=None):
    """
    The exact recursive filter, run on seq [B, L, n] from x = 0: the residuals after `warm`
    frames (the state forgets its start), [B, L - warm - n_delay - 1, n]. The predictions are
    compared with `target` (default seq) n_delay (+ 1) frames later.
    """
    target = seq if target is None else target
    B, L, n = seq.shape
    x = seq.new_zeros(B, var.shape[1])
    preds = []
    for k in range(L - n_delay):
        newest = x @ var.T                                   # x' = A x
        x_pred = torch.cat([newest, x[:, :-n]], dim=1)
        x = x_pred + (seq[:, k] - newest) @ K.T              # x = x' + K (pol - H x')
        preds.append(x @ C.T)                                # p_k
    p = torch.stack(preds, dim=1)
    t = torch.arange(warm + n_delay + 1, L, device=seq.device)
    applied = (1.0 - f_delay) * p[:, t - n_delay]
    if f_delay:
        applied = applied + f_delay * p[:, t - n_delay - 1]
    return target[:, t] - applied


def residual_rms(pol, var, K, C, delay, warm=50, target=None):
    """
    RMS of the residual the predictor leaves, run on pol [frames, n] (numpy in, float out).
    `target`: what it is compared with (default pol; in a simulation, the true turbulence).
    """
    n_delay, f_delay = split_delay(delay)
    seq = torch.as_tensor(np.asarray(pol, float), dtype=torch.float64)[None]
    tgt = None if target is None else torch.as_tensor(np.asarray(target, float), dtype=torch.float64)[None]
    with torch.no_grad():
        res = _residuals(seq, *(torch.as_tensor(np.asarray(m, float), dtype=torch.float64) for m in (var, K, C)),
                         n_delay, f_delay, warm, tgt)
    return float(res.pow(2).mean().sqrt())


def held_out(pol, delay, val_frac=0.3, val_frames=3000, warm=50):
    """The held-out part of pol [frames, n] that train_predictor scores on, with the frames
    before it that the filter settles on: score a predictor with residual_rms(held_out(...), ...)."""
    n_delay, _ = split_delay(delay)
    split = int(round(pol.shape[0] * (1 - val_frac)))
    return pol[max(0, split - warm - n_delay - 1):split + val_frames]


def train_predictor(pol, var, K, C, delay, *, train_var=True, val_frac=0.3, val_frames=3000,
                    taps=None, warm=50, batch=512, lr=3e-3, max_seconds=15.0, max_iters=5000,
                    eval_every=25, patience=20, threads=1, seed=0, verbose=False):
    """
    Train (var, K, C) on pol to minimise the residual, from a starting predictor.

    Parameters
    ----------
    pol : ndarray, shape [frames, n_modes]
        Pseudo open loop (the turbulence + WFS noise), oldest first.
    var, K, C : ndarray
        Starting predictor, shapes [n, N n], [N n, n], [n, N n] (varn_lqg's params).
    delay : float
        Loop delay, frames, may be fractional.
    train_var : bool
        False: var is kept, only K and C are trained (fewer parameters).
    val_frac, val_frames : float, int
        The last val_frac of pol is held out (at most val_frames of it are scored): early
        stopping and the result are chosen on it, with the exact recursive filter.
    taps : int or None
        Length of the impulse response in the training loss. Default: until |F^j| < 1e-6 at
        the start (spectral radius of F), between 50 and 400 frames.
    warm : int
        Frames the recursive filter settles on before the held-out residual counts.
    batch : int
        Frames scored per Adam iteration (all of them if fewer).
    lr, max_seconds, max_iters : Adam's step, and the limits: wall-clock seconds, iterations.
    eval_every, patience : the held-out residual is measured every eval_every iterations; the
        training stops after `patience` measurements without improvement.

    Returns
    -------
    var, K, C : ndarray (float64)
        The best predictor on the held-out frames: the starting one if nothing beat it, or if
        the trained filter is unstable.
    info : dict
        held_out_start, held_out_best (rms, units of pol), iterations, seconds, radius_F, taps.
    """
    t0 = time.monotonic()
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    n_delay, f_delay = split_delay(delay)
    pol = np.asarray(pol, float)
    split = int(round(pol.shape[0] * (1 - val_frac)))
    scale = float(np.std(pol[:split])) or 1.0               # the predictor is linear: invariant
    if taps is None:
        radius = min(max(spectral_radius(var, K), 1e-3), 0.999)
        taps = int(np.clip(np.ceil(np.log(1e-6) / np.log(radius)), 50, 400))
    data = torch.as_tensor(pol[:split] / scale, dtype=torch.float32)
    windows = data.unfold(0, taps, 1)                        # [., n, taps], windows[i] = data[i:i+taps] (a view)
    first = taps + n_delay                                   # first target with p_{t-n} and p_{t-n-1}
    if split - first < 100:
        raise ValueError(f"{pol.shape[0]} frames: too few to train with {taps} taps")
    val = torch.as_tensor(held_out(pol, delay, val_frac, val_frames, warm) / scale, dtype=torch.float64)[None]

    params = [torch.tensor(np.asarray(m, float), dtype=torch.float32) for m in (var, K, C)]
    params[0].requires_grad_(train_var)
    params[1].requires_grad_(True)
    params[2].requires_grad_(True)
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)

    def val_rms(ps):
        with torch.no_grad():
            return float(_residuals(val, *(p.double() for p in ps), n_delay, f_delay, warm)
                         .pow(2).mean().sqrt()) * scale

    start = val_rms(params)
    best, best_params, bad, it = start, [p.detach().clone() for p in params], 0, 0
    all_targets = torch.arange(first, split)
    for it in range(1, max_iters + 1):
        t = all_targets if len(all_targets) <= batch else torch.as_tensor(rng.integers(first, split, batch))
        G = _impulse_response(*params, taps).flip(0)         # flipped: windows hold the oldest first
        applied = (1.0 - f_delay) * torch.einsum("bnj,jmn->bm", windows[t - n_delay - taps + 1], G)
        if f_delay:
            applied = applied + f_delay * torch.einsum("bnj,jmn->bm", windows[t - n_delay - taps], G)
        loss = (data[t] - applied).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % eval_every == 0:
            v = val_rms(params)
            if verbose:
                print(f"  it {it:5d}  batch {float(loss.detach().sqrt()) * scale:.5f}  held-out {v:.5f}")
            if v < best - 1e-12:
                best, best_params, bad = v, [p.detach().clone() for p in params], 0
            else:
                bad += 1
            if bad >= patience or time.monotonic() - t0 > max_seconds:
                break
    out = [p.double().numpy() for p in best_params]
    radius = spectral_radius(out[0], out[1])
    if radius >= 1.0 or not all(np.all(np.isfinite(m)) for m in out):
        out, best = [np.asarray(m, float) for m in (var, K, C)], start   # unstable: keep the start
    return (*out, {"held_out_start": start, "held_out_best": best, "iterations": it,
                   "seconds": time.monotonic() - t0, "radius_F": radius, "taps": taps})
