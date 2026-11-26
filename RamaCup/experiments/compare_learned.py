"""
Compare varn_lqg (model-based: A by least squares, Q from residuals, R, Riccati) with the same
predictor learnt end to end on the telemetry (varn_lqg's learned_predictor.py), on a simulated
loop.

Simulation: n modes, each an AR(2) process (slow, resonant), mixed so that the modes correlate:
the turbulence is exactly a VAR(2), whose true model gives the best linear predictor (the
"oracle"). The WFS adds white noise. pol = turbulence + noise.

1. Offline: fit on `train` frames of pol, score on 10 000 new frames: the rms of the true
   residual (turbulence - what the DM applies), which is what the DM leaves; the measured
   residual adds the noise on top.
2. Closed loop: a 60 s Run at 500 Hz, varn_lqg's controller and optimizer with its config
   "learn" off (the model) and "all" (the learnt predictor), with a stand-in ramacup
   (rama_cup's pseudo open loop and delay blend).

    python compare_learned.py            (about 20 minutes on one core)
"""
import contextlib
import importlib.util
import io
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VARN = HERE.parent / "rama_controller" / "varn_lqg"
sys.path.insert(0, str(VARN))                                # varn_lqg's learned_predictor
from learned_predictor import residual_rms, train_predictor  # noqa: E402

# varn_lqg's modules, with a stand-in for ramacup (only the base classes are used)
_fake = types.ModuleType("ramacup")
_fake.Controller = _fake.Optimizer = object
sys.modules.setdefault("ramacup", _fake)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, VARN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ctl, opt = _load("controller_varn_lqg"), _load("optimizer_varn_lqg")


# ---------------------------------------------------------------- simulation

def make_turbulence(T, n, seed, sigma=0.02):
    """VAR(2) turbulence [T, n] and its true model (var [n, 2n], Q [n, n])."""
    rng = np.random.default_rng(seed)
    a1, a2 = np.linspace(1.85, 1.5, n), np.linspace(-0.9, -0.6, n)
    M = np.eye(n) + 0.3 * rng.standard_normal((n, n)) / np.sqrt(n)
    s = np.zeros((T + 500, n))
    for k in range(2, T + 500):
        s[k] = a1 * s[k - 1] + a2 * s[k - 2] + sigma * rng.standard_normal(n)
    Minv = np.linalg.inv(M)
    var = np.hstack([M @ np.diag(a1) @ Minv, M @ np.diag(a2) @ Minv])
    return s[500:] @ M.T, var, sigma ** 2 * M @ M.T


def predictor_from_model(var, Q_top, R, order, delay):
    """varn_lqg's predictor (var, K, predict) from a model: Riccati and A^delay."""
    n = var.shape[0]
    if var.shape[1] < order * n:                             # pad a VAR(2) model to VAR(order)
        var = np.hstack([var, np.zeros((n, order * n - var.shape[1]))])
    A = opt.companion(var)
    Q = np.zeros_like(A)
    Q[:n, :n] = Q_top
    K, _ = opt.kalman_gain(A, Q, R, 2000)
    return var, K, opt.ahead(A, delay)[:n]


# ---------------------------------------------------------------- 1. offline

def offline(n, noise, delay, order, train, seed=0):
    phi, var_true, Q_true = make_turbulence(train + 10000, n, seed)
    rng = np.random.default_rng(seed + 1)
    pol = phi + noise * rng.standard_normal(phi.shape)
    tr, te, te_phi = pol[:train], pol[train:], phi[train:]
    R_true = noise ** 2 * np.eye(n)

    def score(model):
        return residual_rms(te, *model, delay, target=te_phi)

    rows = [("oracle (true model)", score(predictor_from_model(var_true, Q_true, R_true, order, delay)), None)]
    var_ls, Q_ls = opt.fit_var(tr.T, order)
    starts = [("varn_lqg, R true", R_true), ("varn_lqg, R x 10", 10 * R_true),
              ("varn_lqg, R / 10", R_true / 10), ("varn_lqg, R = 0.3/n (var_lqg)", 0.3 / n * np.eye(n))]
    for label, R in starts:
        start = predictor_from_model(var_ls, Q_ls, R, order, delay)
        t0 = time.monotonic()
        *learned, info = train_predictor(tr, *start, delay)
        rows.append((label, score(start), (score(learned), time.monotonic() - t0, info["iterations"])))
    return rows


# ---------------------------------------------------------------- 2. closed loop

class _DA:
    pass


def closed_loop(kind, n, noise, delay, order, T=30000, seed=0, learn_seconds=15.0):
    """
    Residual rms (true) of a controller on the simulated loop, over the whole run and its last
    third, and the optimizer's log. kind: "integrator", "varn_lqg" (learn off), "learned"
    (learn all), "oracle" (the true model from frame 0). The optimizer keeps its defaults: a fit
    every optimize_every frames, on the last telemetry_depth frames (fewer at the start).
    """
    phi, var_true, Q_true = make_turbulence(T, n, seed)
    rng = np.random.default_rng(seed + 1)
    nd, fd = ctl.split_delay(delay)
    cda = _DA()
    cda.n_modes, cda.delay, cda.delay_exact = n, round(delay), delay
    cda.reset, cda.params_changed, cda.on_params = False, False, (lambda check: None)
    controller = ctl.VarNLQG(order) if kind != "integrator" else None
    if controller:
        controller.setup(cda)
    oda = _DA()
    oda.n_modes, oda.delay, oda.delay_exact, oda.params_changes = n, round(delay), delay, []
    optimizer = opt.VarNLQGFit()
    with contextlib.redirect_stdout(io.StringIO()):
        optimizer.setup(oda)
    oda.config["learn"] = "all" if kind == "learned" else "off"
    oda.config["learn_seconds"] = learn_seconds
    depth, every = oda.config["telemetry_depth"], oda.config["optimize_every"]
    oda.params = getattr(cda, "params", {})
    optimizer.R_input = noise ** 2 * np.eye(n)              # the R files: the true noise
    log = []

    def set_params(changes):
        cda.params.update(changes)
        cda.params_changed = True
    oda.set_params = set_params
    if kind == "oracle":
        set_params(opt.pack(*predictor_from_model(var_true, Q_true, noise ** 2 * np.eye(n), order, delay)))

    outs, inputs, res = np.zeros((T, n)), np.zeros((T, n)), np.zeros((T, n))
    applied_all = np.zeros((T, n))
    u = np.zeros(n)
    for k in range(T):
        applied = np.zeros(n)
        if k >= nd:
            applied += (1 - fd) * outs[k - nd]
        if fd and k >= nd + 1:
            applied += fd * outs[k - nd - 1]
        applied_all[k] = applied
        res[k] = phi[k] + applied
        inputs[k] = res[k] + noise * rng.standard_normal(n)
        if kind == "integrator":
            u = 0.995 * u - 0.3 * inputs[k]
            outs[k] = u
            continue
        cda.input = inputs[k].astype(np.float32)
        controller.step(cda)
        cda.params_changed = False
        outs[k] = cda.output
        if kind != "oracle" and (k + 1) % every == 0:
            lo = max(0, k + 1 - depth)
            tel = _DA()
            tel.input, tel.pol = inputs[lo:k + 1], inputs[lo:k + 1] - applied_all[lo:k + 1]
            tel.dm_limited, tel.seq = np.zeros(k + 1 - lo, bool), np.arange(lo, k + 1)
            tel.current = tel
            oda.telemetry = tel
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                optimizer.optimize(oda)
            log.append(f"frame {k + 1}: {buffer.getvalue().strip()}")

    def rms(r):
        return float(np.sqrt((r ** 2).mean()))
    return rms(res), rms(res[-T // 3:]), log


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    print("1. OFFLINE: fit on `train` frames, true residual rms on 10 000 new frames")
    cases = [dict(n=8, noise=0.05, delay=2.0, order=2, train=2000),
             dict(n=8, noise=0.05, delay=2.3, order=2, train=2000),
             dict(n=8, noise=0.10, delay=2.3, order=2, train=2000),
             dict(n=8, noise=0.05, delay=2.3, order=4, train=2000),
             dict(n=32, noise=0.05, delay=2.3, order=2, train=2000),
             dict(n=32, noise=0.05, delay=2.3, order=2, train=5000)]
    for case in cases:
        print(f"\n{case}")
        print(f"  {'start':32s} {'model-based':>11s} {'learnt':>8s} {'gain':>7s} {'training':>9s}")
        for label, base, learned in offline(**case):
            if learned is None:
                print(f"  {label:32s} {base:11.4f}")
            else:
                r, seconds, its = learned
                print(f"  {label:32s} {base:11.4f} {r:8.4f} {100 * (r - base) / base:+6.1f}% "
                      f"{seconds:7.1f} s ({its} it)")

    print("\n2. CLOSED LOOP: a 60 s Run at 500 Hz (30 000 frames), delay 2.3, VAR(2), the optimizer's "
          "defaults (a fit every 2000 frames, on up to 10 000); true residual rms")
    for n in (8, 96):
        print(f"  {n} modes{'':13s} {'whole run':>9s} {'last third':>10s}")
        for kind in ("integrator", "varn_lqg", "learned", "oracle"):
            whole, last, log = closed_loop(kind, n=n, noise=0.05, delay=2.3, order=2)
            print(f"    {kind:18s} {whole:9.4f} {last:10.4f}")
            if kind == "learned":
                sent = sum(": sent" in line for line in log)
                print(f"      ({len(log)} fits, {sent} learnt predictors sent)")
