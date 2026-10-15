import warnings
import torch
from dataclasses import dataclass
from typing import NamedTuple

@dataclass
class KalmanData:
    wfs_measurements: torch.Tensor
    modal_dm_commands: torch.Tensor
    
@dataclass
class KalmanState:
    x: torch.Tensor
    P: torch.Tensor

    
@dataclass
class KalmanMatrices:
    A: torch.Tensor
    H: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor

def estimate_atmospheric_state(z, u, delay):
    """
    Estimate the atmospheric modal state from closed-loop telemetry.

    Parameters
    ----------
    z : torch.Tensor
        WFS modal measurements, shape [N_frames, N_modes].
    u : torch.Tensor
        Modal DM commands, shape [N_frames, N_modes].

    Returns
    -------
    x : torch.Tensor
        Estimated atmospheric state, x_k ~= z_k + u_k.
        Shape [N_frames, N_modes].
    """
    if z.shape != u.shape:
        raise ValueError(
            f"z and u must have the same shape, got {z.shape} and {u.shape}"
        )
    if delay == 0:
        return z + u

    return z[delay:] + u[:-delay]


def estimate_A(x, method="least_squares"):
    """
    Estimate the state-transition matrix A from:

        x[k+1] = A @ x[k] + w[k]

    Parameters
    ----------
    x : torch.Tensor
        Estimated atmospheric states, shape [N_frames, N_modes].
    method : str
        "least_squares"
            Full matrix, fitted by least squares on consecutive frames.
            Fine at high SNR, but WFS noise in x biases A towards zero
            (the model then underestimates how slowly turbulence evolves).
        "diagonal"
            Same as "least_squares", but each mode evolves independently:
                x_i[k+1] = a_i * x_i[k] + w_i[k]
            Fewer parameters, so less sensitive to short data sets,
            but ignores coupling between modes (e.g. from wind).
        "instrumental_variable"
            Full matrix that is not biased by WFS noise. Uses frame k-1
            as a reference ("instrument"): since the noise is independent
            from frame to frame, it cancels out of
                A = C2 @ inv(C1),
            with C1 = E[x_k x_{k-1}^T] and C2 = E[x_{k+1} x_{k-1}^T].
            Preferred at low SNR; needs no knowledge of R.

    Returns
    -------
    A : torch.Tensor
        State-transition matrix, shape [N_modes, N_modes].
    """
    Xm = x[:-2]   # x_{k-1}
    X = x[1:-1]   # x_k
    Xp = x[2:]    # x_{k+1}

    if method == "least_squares":
        # Solve X @ A.T ~= Xp
        return torch.linalg.lstsq(X, Xp).solution.T

    if method == "diagonal":
        a = torch.sum(Xp * X, dim=0) / torch.sum(X * X, dim=0)
        return torch.diag(a)

    if method == "instrumental_variable":
        # A @ (X.T @ Xm) = Xp.T @ Xm  <=>  (Xm.T @ X) @ A.T = Xm.T @ Xp
        return torch.linalg.solve(Xm.T @ X, Xm.T @ Xp).T

    raise ValueError(
        f"Unknown method {method!r}; use 'least_squares', 'diagonal' "
        "or 'instrumental_variable'"
    )


def estimate_raw_Q(x, A):
    """
    Estimate the covariance of the one-step prediction errors:

        e_k = x[k+1] - A @ x[k]

    Note that this is NOT yet the atmospheric process covariance Q,
    because the state estimate x contains WFS measurement noise.

    Parameters
    ----------
    x : torch.Tensor
        Estimated atmospheric states, shape [N_frames, N_modes].

    A : torch.Tensor
        State-transition matrix, shape [N_modes, N_modes].

    Returns
    -------
    Q_raw : torch.Tensor
        Raw prediction-error covariance.
    """
    X = x[:-1]
    Xp = x[1:]

    prediction_error = Xp - X @ A.T

    return torch.cov(prediction_error.T)

def estimate_Q(Q_raw, A, R, force_positive=False):
    """
    Estimate the atmospheric process covariance Q by removing
    the contribution from WFS measurement noise.

        Q ~= Q_raw - R - A @ R @ A.T

    Parameters
    ----------
    Q_raw : torch.Tensor
        Raw prediction-error covariance.
    A : torch.Tensor
        State-transition matrix.
    R : torch.Tensor
        WFS measurement-noise covariance.
    force_positive : bool
        A covariance cannot have negative eigenvalues, but the subtraction
        above can produce them when R is large compared to Q_raw (low SNR)
        or A is poorly estimated; the Kalman filter can then diverge.
        If True, negative eigenvalues are set to zero (nearest positive
        semi-definite matrix).

    Returns
    -------
    Q : torch.Tensor
        Estimated process covariance.
    """
    Q = Q_raw - R - A @ R @ A.T

    # Numerical symmetrization
    Q = 0.5 * (Q + Q.T)

    if force_positive:
        eigenvalues, eigenvectors = torch.linalg.eigh(Q)
        Q = eigenvectors @ torch.diag(eigenvalues.clamp(min=0)) @ eigenvectors.T

    return Q

def estimate_R(recon, ref_intensity, nPhotons, ron):

    cov_ron = ron**2/nPhotons**2 * recon.T @ recon
    cov_photon = 1/nPhotons * (ref_intensity.reshape(-1,1) * recon).T @ recon

    return (cov_ron + cov_photon).squeeze()


def kalman_filter(
    measurement: torch.Tensor,
    state: KalmanState,
    matrices: KalmanMatrices,
) -> tuple[KalmanState, torch.Tensor]:
    """
    Perform one Kalman filter iteration.

    Parameters
    ----------
    measurement : torch.Tensor
        Current measurement y_k.

    state : KalmanState
        Previous posterior state estimate:
            state.x = x_{k-1|k-1}
            state.P = P_{k-1|k-1}

    matrices : KalmanMatrices
        Kalman filter matrices.

    Returns
    -------
    state : KalmanState
        Updated posterior state:
            state.x = x_{k|k}
            state.P = P_{k|k}

    K : torch.Tensor
        Kalman gain.
    """

    A = matrices.A
    H = matrices.H
    Q = matrices.Q
    R = matrices.R

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    x_pred = A @ state.x

    P_pred = A @ state.P @ A.T + Q

    # ---------------------------------------------------------
    # Measurement update
    # ---------------------------------------------------------

    innovation = measurement - H @ x_pred

    S = H @ P_pred @ H.T + R

    # K = P_pred @ H.T @ inv(S)
    # Use solve rather than explicitly computing inv(S).
    K = torch.linalg.solve(
        S,
        H @ P_pred,
    ).T

    x_new = x_pred + K @ innovation

    # Joseph form for numerical stability
    I = torch.eye(
        P_pred.shape[0],
        device=P_pred.device,
        dtype=P_pred.dtype,
    )

    IKH = I - K @ H

    P_new = (
        IKH @ P_pred @ IKH.T
        + K @ R @ K.T
    )

    return KalmanState(x=x_new, P=P_new), K


# =============================================================
# Steady-state (DARE) gain
# =============================================================

class DAREResult(NamedTuple):
    """
    Solution of the filtering discrete algebraic Riccati equation.

    K : torch.Tensor
        Steady-state Kalman gain, shape [N_state, N_meas].
    P : torch.Tensor
        Steady-state PREDICTED covariance P_{k|k-1}, shape [N_state, N_state].
    converged : bool
        Whether the relative change of P fell below `tol`.
    iterations : int
        Iterations used (doubling steps or Riccati steps).
    residual : float
        Relative DARE residual ||Ric(P) - P||_F / ||P||_F.
    """
    K: torch.Tensor
    P: torch.Tensor
    converged: bool
    iterations: int
    residual: float


def _sym(M):
    return 0.5 * (M + M.transpose(-1, -2))


def _riccati_step(A, H, Q, R, P):
    """One step of the prediction Riccati recursion, P_{k|k-1} -> P_{k+1|k}."""
    S = H @ P @ H.T + R
    K = torch.linalg.solve(S, H @ P).T          # P H^T S^{-1}  (P, S symmetric)
    I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    IKH = I - K @ H
    P_post = IKH @ P @ IKH.T + K @ R @ K.T      # Joseph form
    return _sym(A @ P_post @ A.T + Q)


def steady_state_gain(
    A, H, Q, R,
    tol=1e-6,
    max_iter=None,
    method="doubling",
    compute_dtype=torch.float32,
    on_fail="warn",
):
    """
    Steady-state Kalman gain from the filtering DARE:

        P = A P A^T - A P H^T (H P H^T + R)^{-1} H P A^T + Q
        K = P H^T (H P H^T + R)^{-1}

    P is the predicted covariance P_{k|k-1}, so K is the limit of the gain
    returned by `kalman_filter` when its recursion is iterated.

    Parameters
    ----------
    A : torch.Tensor
        State-transition matrix, shape [N_state, N_state].
    H : torch.Tensor
        Measurement matrix, shape [N_meas, N_state].
    Q : torch.Tensor
        Process-noise covariance, shape [N_state, N_state] (PSD).
    R : torch.Tensor
        Measurement-noise covariance, shape [N_meas, N_meas] (PD for "doubling").
    tol : float
        Convergence threshold on the relative change of P between iterations.
    max_iter : int or None
        Maximum iterations. Default: 64 for "doubling", 10000 for "riccati".
    method : str
        "doubling"
            Structure-preserving doubling algorithm (SDA). Iteration k gives
            the Riccati recursion after 2^k steps, so it converges in
            ~log2(1 / (1 - rho^2)) iterations even when A has eigenvalues
            close to 1. With G = H^T R^{-1} H, A_0 = A^T, G_0 = G, P_0 = Q and
            W_k = I + G_k P_k:
                A_{k+1} = A_k W_k^{-1} A_k
                G_{k+1} = G_k + A_k W_k^{-1} G_k A_k^T
                P_{k+1} = P_k + A_k^T P_k W_k^{-1} A_k
        "riccati"
            Plain iteration of the prediction Riccati recursion from P = Q
            (linear convergence; does not need R^{-1}).
    compute_dtype : torch.dtype
        dtype used internally. The inputs are rescaled by the mean variance of
        Q and R first, so float32 is safe with OPD units (~1e-16 m^2).
    on_fail : str
        "warn", "raise" or "ignore" when the iteration does not converge or
        produces non-finite values.

    Returns
    -------
    DAREResult
        (K, P, converged, iterations, residual), with K and P in the dtype
        and on the device of A.
    """
    out_dtype = A.dtype
    n, m = A.shape[0], H.shape[0]

    # Rescale variances to O(1): K is invariant, P scales with them.
    scale = (torch.diagonal(Q).sum() + torch.diagonal(R).sum()) / (n + m)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).to(compute_dtype)
    A_ = A.to(compute_dtype)
    H_ = H.to(compute_dtype)
    Q_ = _sym(Q.to(compute_dtype) / scale)
    R_ = _sym(R.to(compute_dtype) / scale)

    converged = False
    it = 0

    if method == "doubling":
        max_iter = 64 if max_iter is None else max_iter
        I = torch.eye(n, device=A_.device, dtype=compute_dtype)
        Ak = A_.T.clone()
        Gk = _sym(H_.T @ torch.linalg.solve(R_, H_))
        Pk = Q_.clone()
        for it in range(1, max_iter + 1):
            LU, piv = torch.linalg.lu_factor(I + Gk @ Pk)
            WiA_G = torch.linalg.lu_solve(LU, piv, torch.cat([Ak, Gk], dim=1))
            WiA, WiG = WiA_G[:, :n], WiA_G[:, n:]
            # P_k W_k^{-1} = (W_k^{-T} P_k)^T, since P_k is symmetric
            PWi = torch.linalg.lu_solve(LU, piv, Pk, adjoint=True).T
            P_next = _sym(Pk + Ak.T @ PWi @ Ak)
            Gk = _sym(Gk + Ak @ WiG @ Ak.T)
            Ak = Ak @ WiA
            change = torch.linalg.norm(P_next - Pk) / torch.linalg.norm(P_next)
            Pk = P_next
            if not torch.isfinite(change):
                break
            if change < tol:
                converged = True
                break
        P = Pk

    elif method == "riccati":
        max_iter = 10000 if max_iter is None else max_iter
        P = Q_.clone()
        for it in range(1, max_iter + 1):
            P_next = _riccati_step(A_, H_, Q_, R_, P)
            change = torch.linalg.norm(P_next - P) / torch.linalg.norm(P_next)
            P = P_next
            if not torch.isfinite(change):
                break
            if change < tol:
                converged = True
                break
    else:
        raise ValueError(f"Unknown method {method!r}; use 'doubling' or 'riccati'")

    finite = bool(torch.isfinite(P).all())
    converged = converged and finite
    if finite:
        residual = float(
            torch.linalg.norm(_riccati_step(A_, H_, Q_, R_, P) - P)
            / torch.linalg.norm(P)
        )
    else:
        residual = float("inf")

    S = H_ @ P @ H_.T + R_
    K = torch.linalg.solve(S, H_ @ P).T

    if not converged:
        msg = (
            f"steady_state_gain ({method}) did not converge after {it} "
            f"iterations (relative DARE residual {residual:.2e})"
        )
        if on_fail == "raise":
            raise RuntimeError(msg)
        if on_fail == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return DAREResult(
        K=K.to(out_dtype),
        P=(P * scale).to(out_dtype),
        converged=converged,
        iterations=it,
        residual=residual,
    )


def innovation_whitener(H, P, R):
    """
    Matrix W such that e = W @ nu has identity covariance, for the
    innovation nu_k = y_k - H x_{k|k-1} with covariance

        S = H P H^T + R = L L^T,   W = L^{-1}.

    The normalized innovation squared NIS_k = ||e_k||^2 then has mean N_meas
    when the model is consistent with the data.

    Parameters
    ----------
    H : torch.Tensor
        Measurement matrix, shape [N_meas, N_state].
    P : torch.Tensor
        Predicted covariance P_{k|k-1} (e.g. `DAREResult.P`),
        shape [N_state, N_state].
    R : torch.Tensor
        Measurement-noise covariance, shape [N_meas, N_meas].

    Returns
    -------
    W : torch.Tensor
        Lower-triangular whitening matrix, shape [N_meas, N_meas].
    """
    S = _sym(H @ P @ H.T + R)
    L = torch.linalg.cholesky(S)
    I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    return torch.linalg.solve_triangular(L, I, upper=False)


# =============================================================
# Real-time step
# =============================================================

def build_realtime_matrix(A, H, K, delay, C=None, whiten=None):
    """
    Stack the steady-state filter and the command prediction into a single
    matrix M, so that one matrix-vector product per frame gives

        [x_{k|k}; u_k] = M @ [x_{k-1|k-1}; y_k]

    with
        x_{k|k} = F x_{k-1|k-1} + K y_k,     F = (I - K H) A
        u_k     = C A^d x_{k|k}              (d = delay)

    i.e.
        M = [[ F,          K        ],
             [ C A^d F,    C A^d K  ]]

    Optionally a third block row returns the whitened innovation

        e_k = W (y_k - H A x_{k-1|k-1}) = [-W H A,  W] @ [x_{k-1|k-1}; y_k]

    whose squared norm is the NIS (see `innovation_whitener`).

    Parameters
    ----------
    A : torch.Tensor
        State-transition matrix, shape [N_state, N_state].
    H : torch.Tensor
        Measurement matrix, shape [N_meas, N_state].
    K : torch.Tensor
        Steady-state gain, shape [N_state, N_meas].
    delay : int
        Number of steps d the command is predicted ahead of x_{k|k}.
        With the pseudo-open-loop pairing y_k = z_k + u_{k-2}, x_{k|k}
        estimates phi_{k-1} and u_k acts on phi_{k+1}, so d = 2.
    C : torch.Tensor or None
        Command matrix, shape [N_cmd, N_state]. Default H, which gives
        u_k = A^d x_{k|k} when H = I, and picks the current modes out of an
        augmented (AR(2)) state when H = [I, 0].
    whiten : torch.Tensor or None
        Innovation whitening matrix W, shape [N_meas, N_meas].

    Returns
    -------
    M : torch.Tensor
        Shape [N_state + N_cmd (+ N_meas), N_state + N_meas].
    """
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    C = H if C is None else C

    F = (I - K @ H) @ A
    CAd = C @ torch.linalg.matrix_power(A, delay)

    rows = [
        torch.cat([F, K], dim=1),
        torch.cat([CAd @ F, CAd @ K], dim=1),
    ]
    if whiten is not None:
        rows.append(torch.cat([-whiten @ H @ A, whiten], dim=1))
    return torch.cat(rows, dim=0)


def realtime_step(M, x, y, n_command=None):
    """
    One real-time frame with the stacked matrix from `build_realtime_matrix`:

        [x_{k|k}; u_k (; e_k)] = M @ [x_{k-1|k-1}; y_k]

    Parameters
    ----------
    M : torch.Tensor
        Stacked real-time matrix, shape [N_state + N_out, N_state + N_meas].
    x : torch.Tensor
        Previous posterior state x_{k-1|k-1}, shape [N_state] or [B, N_state].
    y : torch.Tensor
        Pseudo-open-loop measurement y_k, shape [N_meas] or [B, N_meas].
    n_command : int or None
        If given, u is the first `n_command` outputs after the state, and
        any remaining rows (the whitened innovation) are dropped.

    Returns
    -------
    x_new : torch.Tensor
        Posterior state x_{k|k}, same shape as x.
    u : torch.Tensor
        Command u_k (followed by e_k if M has whitening rows and n_command
        is None), shape [N_out] or [B, N_out].
    """
    n = x.shape[-1]
    out = torch.cat([x, y], dim=-1) @ M.T
    x_new = out[..., :n]
    u = out[..., n:] if n_command is None else out[..., n:n + n_command]
    return x_new, u


# =============================================================
# Identification from lagged covariances
# =============================================================
#
# All estimators below work on the lag covariances
#
#     C_l = E[x_{k+l} x_k^T],   l = 0, 1, 2, ...
#
# of the pseudo-open-loop state x_k = s_k + n_k (atmosphere + white WFS
# noise). They can come from a fixed window (`lagged_covariances`) or from
# exponentially-forgotten running sums (`RecursiveCovariance`), so every
# estimator works with both.

def lagged_covariances(x, max_lag=2):
    """
    Sample lag covariances of a window of telemetry:

        C_l = 1 / (N - l) * sum_k x_{k+l} x_k^T,   l = 0 .. max_lag

    Second moments, not centered (the atmospheric modes have zero mean).

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop states, shape [N_frames, N_modes].
    max_lag : int
        Largest lag returned.

    Returns
    -------
    C : list of torch.Tensor
        [C_0, ..., C_max_lag], each shape [N_modes, N_modes].
    """
    N = x.shape[0]
    return [x[l:].T @ x[:N - l] / (N - l) for l in range(max_lag + 1)]


def autocovariances(x, max_lag):
    """
    Per-mode autocovariances (the diagonals of the lag covariances), cheap
    enough to go to large lags:

        gamma_l[i] = 1 / (N - l) * sum_k x_i[k+l] x_i[k],   l = 0 .. max_lag

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop states, shape [N_frames, N_modes].
    max_lag : int
        Largest lag returned.

    Returns
    -------
    gamma : torch.Tensor
        Shape [max_lag + 1, N_modes].
    """
    N = x.shape[0]
    return torch.stack([(x[l:] * x[:N - l]).sum(dim=0) / (N - l) for l in range(max_lag + 1)])


class RecursiveCovariance:
    """
    Lag covariances with a forgetting factor lambda, as an alternative to a
    fixed window:

        S_l <- lambda S_l + (1 - lambda) x_k x_{k-l}^T
        w   <- lambda w   + (1 - lambda)
        C_l = S_l / w

    The effective memory is ~1 / (1 - lambda) frames; dividing by w removes
    the start-up bias. Frames can be fed one at a time or in blocks, with
    the same result. Per-mode autocovariances (`autocovariances`) can be
    tracked to larger lags at O(N_modes) cost per lag, for AR(2) modes.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    forgetting : float
        Forgetting factor lambda in (0, 1).
    max_lag : int
        Largest lag of the full covariance matrices.
    n_autocov_lags : int
        Largest lag of the per-mode autocovariances (0: not tracked).
    device, dtype
        Where to keep the running sums.
    """

    def __init__(self, n_modes, forgetting, max_lag=2, n_autocov_lags=0,
                 device=None, dtype=torch.float32):
        if not 0.0 < forgetting < 1.0:
            raise ValueError(f"forgetting must be in (0, 1), got {forgetting}")
        self.forgetting = forgetting
        self.max_lag = max_lag
        self.n_autocov_lags = n_autocov_lags
        self.S = torch.zeros(max_lag + 1, n_modes, n_modes, device=device, dtype=dtype)
        self.g = torch.zeros(n_autocov_lags + 1, n_modes, device=device, dtype=dtype)
        self.weight = 0.0
        self.n_keep = max(max_lag, n_autocov_lags)
        self.tail = torch.zeros(0, n_modes, device=device, dtype=dtype)  # last n_keep frames

    def update(self, x):
        """
        Add frames, oldest first.

        Parameters
        ----------
        x : torch.Tensor
            New frames, shape [B, N_modes] or [N_modes].
        """
        x = x.reshape(-1, self.S.shape[-1]).to(self.S.dtype)
        B, lam = x.shape[0], self.forgetting
        full = torch.cat([self.tail, x], dim=0)   # history needed for the lags
        t0 = self.tail.shape[0]
        # frame j of the block gets weight (1 - lam) * lam^(B - 1 - j)
        w = (1 - lam) * lam ** torch.arange(B - 1, -1, -1, device=x.device, dtype=x.dtype)
        decay = lam ** B

        def pairs(l):
            j0 = max(0, l - t0)               # first new frame that has a lag-l partner
            return j0, full[t0 + j0:], full[t0 + j0 - l:full.shape[0] - l]

        for l in range(self.max_lag + 1):
            j0, cur, past = pairs(l)
            self.S[l].mul_(decay).add_((cur * w[j0:, None]).T @ past)
        for l in range(self.n_autocov_lags + 1):
            j0, cur, past = pairs(l)
            self.g[l].mul_(decay).add_((cur * past * w[j0:, None]).sum(dim=0))
        self.weight = decay * self.weight + (1 - decay)
        if self.n_keep > 0:
            self.tail = full[-self.n_keep:].clone()

    def covariances(self):
        """
        Returns
        -------
        C : list of torch.Tensor
            [C_0, ..., C_max_lag], each shape [N_modes, N_modes].
        """
        return [S / self.weight for S in self.S]

    def autocovariances(self):
        """
        Returns
        -------
        gamma : torch.Tensor
            Per-mode autocovariances, shape [n_autocov_lags + 1, N_modes].
        """
        return self.g / self.weight


def _solve_right(B, M):
    """B @ inv(M), via a solve."""
    return torch.linalg.solve(M.T, B.T).T


def zernike_radial_groups(n_modes, first_noll=2, device=None):
    """
    Radial order of each Zernike mode, for block-diagonal A.

    Parameters
    ----------
    n_modes : int
        Number of modes, in Noll order.
    first_noll : int
        Noll index of the first mode (2 when piston is excluded, as in
        DeformableMirror.MakeZernikeM2C).

    Returns
    -------
    groups : torch.Tensor
        Radial order n of each mode, shape [N_modes], dtype long.
    """
    j = torch.arange(first_noll, first_noll + n_modes, device=device, dtype=torch.float64)
    n = torch.floor((-1.0 + torch.sqrt(8.0 * (j - 1) + 1)) / 2.0)
    return n.long()


def estimate_A_from_covariances(
    C0, C1, C2=None,
    method="least_squares",
    structure="full",
    groups=None,
    ridge=0.0,
):
    """
    State-transition matrix of x[k+1] = A x[k] + w[k] from lag covariances.

        "least_squares":           A = C1 (C0 + lambda I)^{-1}
        "instrumental_variable":   A = C2 (C1 + lambda I)^{-1}

    Least squares is biased towards zero by WFS noise (C0 = Sigma + R);
    the instrumental variable uses only lags >= 1, where white noise
    cancels (C1 = A Sigma, C2 = A^2 Sigma).

    Parameters
    ----------
    C0, C1, C2 : torch.Tensor
        Lag covariances C_l = E[x_{k+l} x_k^T], shape [N_modes, N_modes].
        C2 is only needed for "instrumental_variable".
    method : str
        "least_squares", "instrumental_variable", or "diagonal"
        (= least squares with structure="diagonal", as in `estimate_A`).
    structure : str
        "full"      dense A, N_modes^2 parameters.
        "diagonal"  a_i = C1_ii / C0_ii  (or C2_ii / C1_ii).
        "block"     block-diagonal A, one block per value of `groups`
                    (e.g. Zernike radial order, see `zernike_radial_groups`);
                    each block is fitted on its own modes only.
    groups : torch.Tensor or None
        Block label of each mode, shape [N_modes]. Required for "block".
    ridge : float
        Tikhonov regularization, relative to the mean variance:
        lambda = ridge * trace(C0) / N_modes.

    Returns
    -------
    A : torch.Tensor
        Shape [N_modes, N_modes].
    """
    if method == "diagonal":
        method, structure = "least_squares", "diagonal"
    if method == "least_squares":
        num, den = C1, C0
    elif method == "instrumental_variable":
        if C2 is None:
            raise ValueError("instrumental_variable needs C2")
        num, den = C2, C1
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'least_squares', "
            "'instrumental_variable' or 'diagonal'"
        )

    n = C0.shape[0]
    lam = ridge * torch.diagonal(C0).sum() / n
    I = torch.eye(n, device=C0.device, dtype=C0.dtype)

    if structure == "full":
        return _solve_right(num, den + lam * I)
    if structure == "diagonal":
        return torch.diag(torch.diagonal(num) / (torch.diagonal(den) + lam))
    if structure == "block":
        if groups is None:
            raise ValueError("structure='block' needs groups")
        A = torch.zeros_like(C0)
        for g in torch.unique(groups):
            idx = torch.nonzero(groups == g).squeeze(1)
            blk = torch.meshgrid(idx, idx, indexing="ij")
            A[blk] = _solve_right(num[blk], den[blk] + lam * I[: len(idx), : len(idx)])
        return A
    raise ValueError(f"Unknown structure {structure!r}; use 'full', 'diagonal' or 'block'")


def apply_structure(M, structure, groups=None):
    """
    Keep only the entries of a covariance allowed by a structure. Zeroing
    entries outside a block (or the diagonal) keeps a PSD matrix PSD.

    With short windows, the off-diagonal entries of an estimated Q or R are
    mostly noise; through K they couple modes that are independent and can
    cost more than the whole gain of the Kalman filter. Their structure
    should match the one used for A.

    Parameters
    ----------
    M : torch.Tensor
        Shape [N, N].
    structure : str
        "full", "diagonal" or "block".
    groups : torch.Tensor or None
        Block label of each row, shape [N] (for "block").

    Returns
    -------
    M_s : torch.Tensor
        Shape [N, N].
    """
    if structure == "full":
        return M
    if structure == "diagonal":
        return torch.diag(torch.diagonal(M))
    if structure == "block":
        if groups is None:
            raise ValueError("structure='block' needs groups")
        g = torch.as_tensor(groups, device=M.device)
        return M * (g[:, None] == g[None, :]).to(M.dtype)
    raise ValueError(f"Unknown structure {structure!r}; use 'full', 'diagonal' or 'block'")


def _clip_psd(M, floor=0.0):
    eigenvalues, eigenvectors = torch.linalg.eigh(_sym(M))
    return eigenvectors @ torch.diag(eigenvalues.clamp(min=floor)) @ eigenvectors.T


def limit_spectral_radius(A, max_radius, structure="full"):
    """
    Shrink A so that its spectral radius is at most `max_radius`.

    Short-window estimates of slow modes (a ~ 0.995 at 1 kHz) overshoot 1
    regularly; the closed-loop cost is flat in a near 1, so shrinking is
    much better than rejecting the whole model.

        "diagonal"      a_i <- clamp(a_i, -max_radius, max_radius)
        "full"/"block"  A <- A * max_radius / rho(A)   if rho(A) > max_radius
                        (for "block", per block)

    Parameters
    ----------
    A : torch.Tensor
        Shape [N_modes, N_modes].
    max_radius : float
        Largest allowed spectral radius.
    structure : str
        "full", "diagonal" or "block" (then A is block diagonal and each
        block is scaled on its own; blocks are found from its sparsity).

    Returns
    -------
    A_s : torch.Tensor
        Shape [N_modes, N_modes].
    """
    if structure == "diagonal":
        return torch.diag(torch.diagonal(A).clamp(-max_radius, max_radius))
    rho = torch.linalg.eigvals(A).abs().max().real.to(A.dtype)
    if structure == "full" or rho <= max_radius:
        return A * torch.clamp(max_radius / rho, max=1.0)
    # block: connected components of the sparsity pattern
    n = A.shape[0]
    label = torch.arange(n, device=A.device)
    adj = (A != 0) | (A.T != 0)
    for _ in range(n):
        new = torch.where(adj, label[None, :], n).min(dim=1).values.minimum(label)
        if torch.equal(new, label):
            break
        label = new
    A = A.clone()
    for g in torch.unique(label):
        idx = torch.nonzero(label == g).squeeze(1)
        blk = torch.meshgrid(idx, idx, indexing="ij")
        rho_g = torch.linalg.eigvals(A[blk]).abs().max().real.to(A.dtype)
        A[blk] = A[blk] * torch.clamp(max_radius / rho_g, max=1.0)
    return A


def estimate_R_from_covariances(C0, C1, A, C2=None, structure="full", min_fraction=1e-4,
                                groups=None):
    """
    WFS noise covariance from the lag covariances of the pseudo-open-loop
    state, assuming x_k = s_k + n_k with s an AR(1) process
    s_{k+1} = A s_k + w_k and n white with covariance R.

    Without C2 ("lag-1" form):

        C0 = Sigma + R,   C1 = A Sigma   =>   R = C0 - A^{-1} C1

    With C2 ("residual" form, recommended): the one-step prediction error
    e_k = x_{k+1} - A x_k = w_k + n_{k+1} - A n_k has lag-1 covariance

        L1 = E[e_k e_{k-1}^T] = C1 - C2 A^T - A C0 + A C1 A^T = -A R
        =>   R = -A^{-1} L1

    Both are identical when A is the full instrumental-variable estimate
    (C2 = A C1). With any other A (diagonal, ridge, block, or just its
    estimation error), the lag-1 form has an error ~ Sigma * dA, while the
    residual form has an error ~ Sigma * dA * (1 - a^2): for slow modes
    (a ~ 0.99), the residual form is ~50x less sensitive to errors in A.

    Needs no photon-noise model, so it also holds for a CNN-linearized WFS
    as long as its error is white in time. Only meaningful for temporally
    correlated modes: where A ~ 0, signal and noise are both white.

    Parameters
    ----------
    C0, C1 : torch.Tensor
        Lag covariances, shape [N_modes, N_modes].
    A : torch.Tensor
        State-transition matrix, shape [N_modes, N_modes].
    C2 : torch.Tensor or None
        Lag-2 covariance; selects the residual form.
    structure : str
        "full"      symmetrized R, eigenvalues clipped to the floor.
        "diagonal"  diagonal of the full estimate, clipped to [floor, C0_ii].
        "block"     blocks of the full estimate (see `apply_structure`),
                    eigenvalues clipped to the floor.
    min_fraction : float
        Floor on the noise, relative to the mean mode variance, so that R
        stays positive definite (the DARE needs R^{-1}): r_i (or the
        eigenvalues of R) >= min_fraction * trace(C0) / N_modes. Relative to
        the mean rather than per mode, because high-SNR modes (tip/tilt)
        can have r_i / C0_ii well below 1e-3.
    groups : torch.Tensor or None
        Block label of each mode, shape [N_modes] (for "block").

    Returns
    -------
    R : torch.Tensor
        Shape [N_modes, N_modes].
    """
    if C2 is None:
        R = C0 - torch.linalg.solve(A, C1)
    else:
        L1 = C1 - C2 @ A.T - A @ C0 + A @ C1 @ A.T
        R = -torch.linalg.solve(A, L1)
    R = _sym(R)

    floor = min_fraction * torch.diagonal(C0).sum() / C0.shape[0]
    if structure == "diagonal":
        r = torch.minimum(torch.diagonal(R), torch.diagonal(C0)).clamp(min=floor)
        return torch.diag(r)
    return _clip_psd(apply_structure(R, structure, groups), floor)


def estimate_R_from_telemetry(x, A, structure="full", min_fraction=1e-4, form="residual"):
    """
    WFS noise covariance from telemetry (see `estimate_R_from_covariances`):

        "lag1":      R = C0 - A^{-1} C1
        "residual":  R = -A^{-1} E[e_k e_{k-1}^T],  e_k = x_{k+1} - A x_k

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop states, shape [N_frames, N_modes].
    A : torch.Tensor
        State-transition matrix, shape [N_modes, N_modes].
    structure : str
        "full" or "diagonal".
    min_fraction : float
        Noise floor (see `estimate_R_from_covariances`).
    form : str
        "residual" (default, robust to errors in A) or "lag1".

    Returns
    -------
    R : torch.Tensor
        Shape [N_modes, N_modes].
    """
    if form not in ("residual", "lag1"):
        raise ValueError(f"Unknown form {form!r}; use 'residual' or 'lag1'")
    C0, C1, C2 = lagged_covariances(x, max_lag=2)
    return estimate_R_from_covariances(
        C0, C1, A, C2=C2 if form == "residual" else None,
        structure=structure, min_fraction=min_fraction,
    )


def estimate_Q_from_covariances(C0, C1, A, R, force_positive=True):
    """
    Process-noise covariance from lag covariances. The raw one-step
    prediction-error covariance

        Q_raw = E[(x_{k+1} - A x_k)(x_{k+1} - A x_k)^T]
              = C0 - A C1^T - C1 A^T + A C0 A^T

    is corrected for WFS noise as in `estimate_Q`:

        Q = Q_raw - R - A R A^T

    Parameters
    ----------
    C0, C1 : torch.Tensor
        Lag covariances, shape [N_modes, N_modes].
    A : torch.Tensor
        State-transition matrix, shape [N_modes, N_modes].
    R : torch.Tensor
        WFS noise covariance, shape [N_modes, N_modes].
    force_positive : bool
        Clip negative eigenvalues (see `estimate_Q`).

    Returns
    -------
    Q : torch.Tensor
        Shape [N_modes, N_modes].
    """
    Q_raw = C0 - A @ C1.T - C1 @ A.T + A @ C0 @ A.T
    return estimate_Q(_sym(Q_raw), A, R, force_positive=force_positive)


class ModelCheck(NamedTuple):
    """
    Result of `validate_model`.

    ok : bool
        True when every check passed.
    reasons : tuple of str
        One entry per failed check.
    spectral_radius : float
        max |eig(A)|.
    negative_Q_fraction : float
        sum of |negative eigenvalues of Q| / sum of |eigenvalues of Q|.
    dare_converged : bool or None
        From the DARE result, if one was given.
    """
    ok: bool
    reasons: tuple
    spectral_radius: float
    negative_Q_fraction: float
    dare_converged: bool


def validate_model(
    A, Q, R,
    dare=None,
    max_spectral_radius=0.9999,
    max_negative_Q_fraction=0.2,
    min_gain=None,
):
    """
    Sanity checks before a re-identified model replaces the running one:

        - all matrices finite
        - spectral radius of A < max_spectral_radius (stable AR model)
        - Q approximately PSD: the negative part of its spectrum, before any
          clipping, is at most `max_negative_Q_fraction` of the total
        - R positive definite
        - the DARE converged (if `dare` is given)
        - every measured mode has K_ii >= min_gain (if `dare` and `min_gain`
          are given). A mode with K_ii ~ 0 (Q_ii clipped to 0) is never
          corrected: its command drifts open loop.

    Parameters
    ----------
    A : torch.Tensor
        State-transition matrix, shape [N_state, N_state].
    Q : torch.Tensor
        Process-noise covariance BEFORE clipping negative eigenvalues,
        shape [N_state, N_state].
    R : torch.Tensor
        WFS noise covariance, shape [N_meas, N_meas].
    dare : DAREResult or None
        Output of `steady_state_gain` for this model.
    max_spectral_radius : float
        Upper bound on max |eig(A)|.
    max_negative_Q_fraction : float
        Upper bound on the negative spectral mass of Q.
    min_gain : float or None
        Lower bound on K_ii for the measured modes, i < N_meas (the state
        starts with the measured modes: H = I or [I, 0]).

    Returns
    -------
    ModelCheck
    """
    reasons = []
    finite = all(bool(torch.isfinite(M).all()) for M in (A, Q, R))
    if not finite:
        reasons.append("non-finite matrix")
        return ModelCheck(False, tuple(reasons), float("nan"), float("nan"),
                          None if dare is None else dare.converged)

    rho = float(torch.linalg.eigvals(A).abs().max())
    if rho >= max_spectral_radius:
        reasons.append(f"spectral radius of A = {rho:.5f} >= {max_spectral_radius}")

    q = torch.linalg.eigvalsh(_sym(Q))
    neg = float(q.clamp(max=0).abs().sum() / q.abs().sum().clamp(min=torch.finfo(q.dtype).tiny))
    if neg > max_negative_Q_fraction:
        reasons.append(f"Q has {100 * neg:.1f}% negative spectral mass")

    if float(torch.linalg.eigvalsh(_sym(R)).min()) <= 0:
        reasons.append("R is not positive definite")

    dare_converged = None
    if dare is not None:
        dare_converged = dare.converged
        if not dare.converged:
            reasons.append(f"DARE did not converge (residual {dare.residual:.2e})")
        if min_gain is not None:
            m = dare.K.shape[1]
            low = torch.nonzero(torch.diagonal(dare.K[:m, :m]) < min_gain).flatten().tolist()
            if low:
                reasons.append(f"K_ii < {min_gain} in modes {low}")

    return ModelCheck(len(reasons) == 0, tuple(reasons), rho, neg, dare_converged)


# =============================================================
# AR(2) state augmentation (vibrations)
# =============================================================
#
# For a set S of s modes (any modes, e.g. tip and tilt), the state is
# augmented with their previous value:
#
#     X_k = [x_k; x_{k-1}[S]],   shape [N_modes + s]
#
# and each mode i in S follows its own AR(2) model
#
#     x_i[k+1] = a1_i x_i[k] + a2_i x_i[k-1] + w_i[k]
#
# A lightly damped resonance at frequency f (sampling fs) with pole radius
# rho has a1 = 2 rho cos(2 pi f / fs), a2 = -rho^2.

def _as_index(modes, device):
    return torch.as_tensor(modes, device=device, dtype=torch.long).reshape(-1)


def estimate_ar2_from_autocovariances(gamma, method="least_squares"):
    """
    Per-mode AR(2) coefficients from autocovariances gamma_l.

        "least_squares" (Yule-Walker):
            [gamma_0  gamma_1] [a1]   [gamma_1]
            [gamma_1  gamma_0] [a2] = [gamma_2]
            Biased by WFS noise, which only enters gamma_0.
        "instrumental_variable" (overdetermined Yule-Walker):
            gamma_{l+1} = a1 gamma_l + a2 gamma_{l-1},   l = 2 .. L-1
            solved in the least-squares sense. Uses lags >= 1 only, so
            white noise cancels. With only l = 2, 3 the system is nearly
            singular for low-frequency resonances (gamma_1 ~ gamma_2 ~
            gamma_3); lags covering a good part of the vibration period
            (L ~ 20 at 1 kHz for f ~ 10-50 Hz) fix the conditioning.

    Parameters
    ----------
    gamma : torch.Tensor
        Autocovariances, shape [L + 1, s] (from `autocovariances` or
        `RecursiveCovariance.autocovariances`); L >= 2 for
        "least_squares", L >= 4 for "instrumental_variable".
    method : str
        "least_squares" or "instrumental_variable".

    Returns
    -------
    a1, a2 : torch.Tensor
        Shape [s] each.
    """
    L = gamma.shape[0] - 1
    if method == "least_squares":
        if L < 2:
            raise ValueError("least_squares needs autocovariances up to lag 2")
        g0, g1, g2 = gamma[0], gamma[1], gamma[2]
        G = torch.stack([torch.stack([g0, g1], -1), torch.stack([g1, g0], -1)], -2)
        a = torch.linalg.solve(G, torch.stack([g1, g2], -1))
    elif method == "instrumental_variable":
        if L < 4:
            raise ValueError("instrumental_variable needs autocovariances up to lag 4")
        l = torch.arange(2, L, device=gamma.device)
        G = torch.stack([gamma[l], gamma[l - 1]], -1).transpose(0, 1)   # [s, n_eq, 2]
        b = gamma[l + 1].T.unsqueeze(-1)                                  # [s, n_eq, 1]
        # normal equations: tiny batched 2x2 solves (lstsq on CUDA needs full rank)
        a = torch.linalg.solve(G.transpose(1, 2) @ G, G.transpose(1, 2) @ b).squeeze(-1)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'least_squares' or 'instrumental_variable'")
    return a[:, 0], a[:, 1]


def estimate_ar2(x, modes, method="least_squares", max_lag=20):
    """
    Per-mode AR(2) coefficients from telemetry,

        x_i[k+1] = a1_i x_i[k] + a2_i x_i[k-1] + w_i[k].

    "least_squares" is the regression of x[k+1] on (x[k], x[k-1]) over the
    window (exact for a noise-free damped sinusoid). "instrumental_variable"
    uses the overdetermined Yule-Walker equations of
    `estimate_ar2_from_autocovariances` up to `max_lag`.

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop states, shape [N_frames, N_modes].
    modes : sequence of int or torch.Tensor
        Indices of the AR(2) modes, length s.
    method : str
        "least_squares" or "instrumental_variable".
    max_lag : int
        Largest lag for "instrumental_variable".

    Returns
    -------
    a1, a2 : torch.Tensor
        Shape [s] each.
    """
    modes = _as_index(modes, x.device)
    xs = x[:, modes]
    if method == "least_squares":
        Xp, X1, X2 = xs[2:], xs[1:-1], xs[:-2]
        G = torch.stack([
            torch.stack([(X1 * X1).sum(0), (X1 * X2).sum(0)], -1),
            torch.stack([(X1 * X2).sum(0), (X2 * X2).sum(0)], -1),
        ], -2)
        b = torch.stack([(Xp * X1).sum(0), (Xp * X2).sum(0)], -1)
        a = torch.linalg.solve(G, b)
        return a[:, 0], a[:, 1]
    return estimate_ar2_from_autocovariances(autocovariances(xs, max_lag), method)


def ar2_noise_variance(gamma, a1, a2):
    """
    WFS noise variance of AR(2) modes from the autocovariance of the
    one-step prediction error e_k = x_{k+1} - a1 x_k - a2 x_{k-1}
    = w_k + n_{k+1} - a1 n_k - a2 n_{k-1}:

        E[e_k e_{k-1}] = -a1 (1 - a2) r,     E[e_k e_{k-2}] = -a2 r

    and r is their least-squares combination. These are computed from
    gamma as sum_ij c_i c_j gamma_{|l + j - i|}, c = [1, -a1, -a2].
    Working on the (small) prediction error avoids subtracting two nearly
    equal large numbers, as gamma_0 - (signal variance) would for a
    lightly damped vibration.

    Parameters
    ----------
    gamma : torch.Tensor
        Autocovariances, shape [L + 1, s], L >= 4.
    a1, a2 : torch.Tensor
        AR(2) coefficients, shape [s].

    Returns
    -------
    r : torch.Tensor
        Noise variances clipped to [0, gamma_0], shape [s].
    """
    if gamma.shape[0] < 5:
        raise ValueError("ar2_noise_variance needs autocovariances up to lag 4")
    c = [torch.ones_like(a1), -a1, -a2]

    def e_cov(l):
        return sum(c[i] * c[j] * gamma[abs(l + j - i)] for i in range(3) for j in range(3))

    k1, k2 = -a1 * (1 - a2), -a2
    r = (k1 * e_cov(1) + k2 * e_cov(2)) / (k1**2 + k2**2)
    return torch.minimum(r.clamp(min=0), gamma[0])


def augment_ar2(A, H, modes, a1, a2):
    """
    Augmented transition and measurement matrices for the state
    X_k = [x_k; x_{k-1}[S]]:

        A_aug = [[A',   A2 ],      A' = A with the rows of S replaced by
                 [E_S,  0  ]]           a1_i on the diagonal,
                                   A2[i, j] = a2_i for mode i = S[j],
        H_aug = [H, 0]             E_S selects the modes S (shift register).

    Columns of A for the modes in S are kept, so other modes can still
    depend on x_k[S].

    Parameters
    ----------
    A : torch.Tensor
        AR(1) transition matrix, shape [N_modes, N_modes].
    H : torch.Tensor
        Measurement matrix, shape [N_meas, N_modes].
    modes : sequence of int or torch.Tensor
        Indices of the AR(2) modes, length s.
    a1, a2 : torch.Tensor
        AR(2) coefficients, shape [s].

    Returns
    -------
    A_aug : torch.Tensor
        Shape [N_modes + s, N_modes + s].
    H_aug : torch.Tensor
        Shape [N_meas, N_modes + s].
    """
    modes = _as_index(modes, A.device)
    n, s = A.shape[0], modes.numel()
    j = torch.arange(s, device=A.device)

    A_aug = torch.zeros(n + s, n + s, device=A.device, dtype=A.dtype)
    A_aug[:n, :n] = A
    A_aug[modes, :] = 0
    A_aug[modes, modes] = a1.to(A.dtype)
    A_aug[modes, n + j] = a2.to(A.dtype)
    A_aug[n + j, modes] = 1

    H_aug = torch.cat([H, torch.zeros(H.shape[0], s, device=H.device, dtype=H.dtype)], dim=1)
    return A_aug, H_aug


def estimate_Q_ar2_from_covariances(C, A_aug, modes, R, force_positive=True):
    """
    Process-noise covariance of the AR(2)-augmented model. With
    x_{k+1} = A1 x_k + A2 x_{k-1} + w_k (A1 = A_aug[:n, :n], A2 the
    [n, n] matrix with the a2 column block scattered onto the modes S),
    B = [A1, A2] and z_k = [x_k; x_{k-1}]:

        Q_raw = C0 - B [C1, C2]^T - [C1, C2] B^T + B Sigma_z B^T,
        Sigma_z = [[C0, C1], [C1^T, C0]]

        Q = Q_raw - R - A1 R A1^T - A2 R A2^T

        Q_aug = [[Q, 0], [0, 0]]    (the shift register is noise-free)

    Parameters
    ----------
    C : list of torch.Tensor
        Lag covariances [C_0, C_1, C_2, ...], each [N_modes, N_modes].
    A_aug : torch.Tensor
        From `augment_ar2`, shape [N_modes + s, N_modes + s].
    modes : sequence of int or torch.Tensor
        Indices of the AR(2) modes, length s.
    R : torch.Tensor
        WFS noise covariance, shape [N_modes, N_modes].
    force_positive : bool
        Clip negative eigenvalues of Q (see `estimate_Q`).

    Returns
    -------
    Q_aug : torch.Tensor
        Shape [N_modes + s, N_modes + s].
    """
    modes = _as_index(modes, A_aug.device)
    n = C[0].shape[0]
    s = modes.numel()
    C0, C1, C2 = C[0], C[1], C[2]

    A1 = A_aug[:n, :n]
    A2 = torch.zeros_like(A1)
    A2[:, modes] = A_aug[:n, n:]
    B = torch.cat([A1, A2], dim=1)
    Cx = torch.cat([C1, C2], dim=1)
    Sz = torch.cat([torch.cat([C0, C1], dim=1), torch.cat([C1.T, C0], dim=1)], dim=0)

    Q_raw = _sym(C0 - B @ Cx.T - Cx @ B.T + B @ Sz @ B.T)
    Q = _sym(Q_raw - R - A1 @ R @ A1.T - A2 @ R @ A2.T)
    if force_positive:
        eigenvalues, eigenvectors = torch.linalg.eigh(Q)
        Q = eigenvectors @ torch.diag(eigenvalues.clamp(min=0)) @ eigenvectors.T

    Q_aug = torch.zeros(n + s, n + s, device=Q.device, dtype=Q.dtype)
    Q_aug[:n, :n] = Q
    return Q_aug


def estimate_Q_ar2(x, A_aug, modes, R, force_positive=True):
    """
    Process-noise covariance of the AR(2)-augmented model from telemetry
    (see `estimate_Q_ar2_from_covariances`).

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop states, shape [N_frames, N_modes].
    A_aug : torch.Tensor
        Shape [N_modes + s, N_modes + s].
    modes : sequence of int or torch.Tensor
        Indices of the AR(2) modes, length s.
    R : torch.Tensor
        WFS noise covariance, shape [N_modes, N_modes].
    force_positive : bool
        Clip negative eigenvalues of Q.

    Returns
    -------
    Q_aug : torch.Tensor
        Shape [N_modes + s, N_modes + s].
    """
    return estimate_Q_ar2_from_covariances(
        lagged_covariances(x, max_lag=2), A_aug, modes, R, force_positive
    )


def ar2_state(x_k, x_km1, modes):
    """
    Augmented state [x_k; x_{k-1}[S]].

    Parameters
    ----------
    x_k, x_km1 : torch.Tensor
        Current and previous states, shape [N_modes] or [B, N_modes].
    modes : sequence of int or torch.Tensor
        Indices of the AR(2) modes, length s.

    Returns
    -------
    X : torch.Tensor
        Shape [N_modes + s] or [B, N_modes + s].
    """
    modes = _as_index(modes, x_k.device)
    return torch.cat([x_k, x_km1[..., modes]], dim=-1)


# =============================================================
# Full identification from lag covariances
# =============================================================

class IdentifiedModel(NamedTuple):
    """
    Output of `identify_model`.

    A, H, Q, R : torch.Tensor
        Model for the Kalman filter (augmented if AR(2) modes are used):
        A [N_state, N_state], H [N_meas, N_state], Q [N_state, N_state]
        (negative eigenvalues clipped), R [N_meas, N_meas].
    Q_unclipped : torch.Tensor
        Q before clipping, for `validate_model`.
    A_ar1 : torch.Tensor
        The AR(1) matrix before augmentation, [N_modes, N_modes].
    """
    A: torch.Tensor
    H: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    Q_unclipped: torch.Tensor
    A_ar1: torch.Tensor


def identify_model(
    C,
    A_method="instrumental_variable",
    A_structure="full",
    groups=None,
    ridge=0.0,
    A_max_radius=None,
    R=None,
    R_structure="match",
    Q_structure="match",
    R_min_fraction=1e-4,
    ar2_modes=None,
    ar2_method="instrumental_variable",
    autocov=None,
):
    """
    Identify (A, Q, R) from lag covariances of pseudo-open-loop telemetry:

        1. A with `estimate_A_from_covariances` (optionally shrunk to
           spectral radius A_max_radius, `limit_spectral_radius`)
        2. R = -A^{-1} E[e_k e_{k-1}^T] (`estimate_R_from_covariances`,
           residual form), unless R is given
        3. optional AR(2) modes: a1, a2 per mode
           (`estimate_ar2_from_autocovariances`), R_ii from
           `ar2_noise_variance` (cross terms of those modes set to 0),
           augmented A and H (`augment_ar2`)
        4. Q = Q_raw - R - A R A^T (AR(1) or AR(2) form), restricted to
           Q_structure, negative eigenvalues clipped

    Parameters
    ----------
    C : list of torch.Tensor
        Lag covariances [C_0, C_1, C_2, ...], each [N_modes, N_modes], from
        `lagged_covariances` or `RecursiveCovariance`.
    A_method, A_structure, groups, ridge
        Passed to `estimate_A_from_covariances`.
    A_max_radius : float or None
        If set, shrink A to this spectral radius before estimating R and Q.
    R : torch.Tensor or None
        Known noise covariance [N_modes, N_modes]; estimated from C if None.
    R_structure, Q_structure : str
        "full", "diagonal", "block" or "match" (same as A_structure), see
        `apply_structure`. Off-diagonal terms estimated from short windows
        are mostly noise and couple independent modes through K, so these
        should not be richer than the structure of A.
    R_min_fraction : float
        Noise floor for the telemetry estimate of R.
    ar2_modes : sequence of int, torch.Tensor or None
        Modes with an AR(2) model (e.g. [0, 1] for tip and tilt).
    ar2_method : str
        Passed to `estimate_ar2_from_autocovariances`.
    autocov : torch.Tensor or None
        Per-mode autocovariances [L + 1, N_modes] (`autocovariances` or
        `RecursiveCovariance.autocovariances`), L >= 4, for the AR(2) modes.
        Default: the diagonals of C.

    Returns
    -------
    IdentifiedModel
    """
    C0, C1, C2 = C[0], C[1], C[2]
    n = C0.shape[0]
    if R_structure == "match":
        R_structure = A_structure
    if Q_structure == "match":
        Q_structure = A_structure
    A = estimate_A_from_covariances(C0, C1, C2, method=A_method, structure=A_structure,
                                    groups=groups, ridge=ridge)
    if A_max_radius is not None:
        A = limit_spectral_radius(A, A_max_radius, A_structure)
    R_from_telemetry = R is None
    if R_from_telemetry:
        R = estimate_R_from_covariances(C0, C1, A, C2=C2, structure=R_structure,
                                        min_fraction=R_min_fraction, groups=groups)
    H = torch.eye(n, device=C0.device, dtype=C0.dtype)

    if ar2_modes is None or len(ar2_modes) == 0:
        Q_raw = C0 - A @ C1.T - C1 @ A.T + A @ C0 @ A.T
        Q_un = apply_structure(estimate_Q(_sym(Q_raw), A, R), Q_structure, groups)
        return IdentifiedModel(A, H, _clip_psd(Q_un), R, Q_un, A)

    modes = _as_index(ar2_modes, C0.device)
    if autocov is None:
        autocov = torch.stack([torch.diagonal(Cl) for Cl in C])
    gamma = autocov[:, modes]
    a1, a2 = estimate_ar2_from_autocovariances(gamma, method=ar2_method)
    if R_from_telemetry:
        r = ar2_noise_variance(gamma, a1, a2)
        r = r.clamp(min=R_min_fraction * torch.diagonal(C0).sum() / n)
        R = R.clone()
        R[modes, :] = 0
        R[:, modes] = 0
        R[modes, modes] = r
    A_aug, H_aug = augment_ar2(A, H, modes, a1, a2)
    Q_un = estimate_Q_ar2_from_covariances(C, A_aug, modes, R, force_positive=False)
    Q_un[:n, :n] = apply_structure(Q_un[:n, :n], Q_structure, groups)
    return IdentifiedModel(A_aug, H_aug, _clip_psd(Q_un), R, Q_un, A)


# =============================================================
# Per-mode AR(p) models from prediction errors
# =============================================================
#
# Each mode i follows its own AR(p) model
#
#     x_i[k+1] = c_i1 x_i[k] + c_i2 x_i[k-1] + ... + c_ip x_i[k-p+1] + w_i[k]
#
# fitted by least squares on a window of pseudo-open-loop telemetry. Q is the
# variance of the actual one-step prediction errors, so it is positive by
# construction, unlike Q_raw - R - A R A^T from lag covariances, which goes
# negative for drifting modes and collapses their gain to K_ii = 0. With a
# real WFS, the telemetry estimate of R is unreliable (correlated innovations,
# unknown units), so R is set relative to Q (R_ii = R_fraction * Q_ii).

def estimate_ar(x, order):
    """
    Per-mode AR(p) coefficients by least squares, and the prediction errors:

        x_i[k+1] = sum_j c_ij x_i[k-j+1] + e_i[k],   j = 1 .. p

    Solved with a batched QR (one p x p problem per mode), which is better
    conditioned than the normal equations for smooth, strongly correlated
    lags.

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop telemetry, shape [N_frames, N_modes].
    order : int
        p >= 1.

    Returns
    -------
    coef : torch.Tensor
        c_ij, shape [N_modes, p] (column j-1 multiplies x[k-j+1]).
    e : torch.Tensor
        One-step prediction errors, shape [N_frames - p, N_modes].
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    N = x.shape[0]
    if N <= 2 * order:
        raise ValueError(f"need more than {2 * order} frames for an AR({order}) fit")
    # X[i, k, j] = x_i[k + p - 1 - j], target t[i, k] = x_i[k + p]
    X = torch.stack([x[order - 1 - j: N - 1 - j] for j in range(order)], dim=-1).transpose(0, 1)
    t = x[order:].T.unsqueeze(-1)
    Qm, Rm = torch.linalg.qr(X)
    coef = torch.linalg.solve_triangular(Rm, Qm.transpose(1, 2) @ t, upper=True).squeeze(-1)
    return coef, ar_prediction_errors(x, coef)


def ar_prediction_errors(x, coef):
    """
    One-step prediction errors of per-mode AR(p) models.

    Parameters
    ----------
    x : torch.Tensor
        Shape [N_frames, N_modes].
    coef : torch.Tensor
        Shape [N_modes, p].

    Returns
    -------
    e : torch.Tensor
        e[k] = x[k + p] - sum_j c_j x[k + p - j], shape [N_frames - p, N_modes].
    """
    p, N = coef.shape[1], x.shape[0]
    pred = sum(coef[:, j] * x[p - 1 - j: N - 1 - j] for j in range(p))
    return x[p:] - pred


def ar_spectral_radius(coef):
    """
    Largest root modulus of each mode's AR polynomial
    z^p - c_1 z^{p-1} - ... - c_p.

    Parameters
    ----------
    coef : torch.Tensor
        Shape [N_modes, p].

    Returns
    -------
    rho : torch.Tensor
        Shape [N_modes].
    """
    n, p = coef.shape
    comp = torch.zeros(n, p, p, device=coef.device, dtype=coef.dtype)
    comp[:, 0, :] = coef
    if p > 1:
        idx = torch.arange(p - 1, device=coef.device)
        comp[:, idx + 1, idx] = 1
    return torch.linalg.eigvals(comp).abs().amax(dim=-1).to(coef.dtype)


def limit_ar_radius(coef, max_radius):
    """
    Shrink the roots of each mode's AR polynomial to modulus <= max_radius:
    scaling every root by r is c_j -> c_j r^j.

    Parameters
    ----------
    coef : torch.Tensor
        Shape [N_modes, p].
    max_radius : float

    Returns
    -------
    coef : torch.Tensor
        Shape [N_modes, p].
    """
    rho = ar_spectral_radius(coef)
    r = torch.clamp(max_radius / rho, max=1.0)
    j = torch.arange(1, coef.shape[1] + 1, device=coef.device, dtype=coef.dtype)
    return coef * r[:, None] ** j[None, :]


def ar_companion(coef):
    """
    State-space form of per-mode AR(p) models, with the stacked state
    X_k = [x_k; x_{k-1}; ...; x_{k-p+1}]:

        A = [[diag(c_1), diag(c_2), ..., diag(c_p)],
             [I,         0,         ..., 0        ],
             ...
             [0,         ...,       I,   0        ]]
        H = [I, 0, ..., 0]

    Parameters
    ----------
    coef : torch.Tensor
        Shape [N_modes, p].

    Returns
    -------
    A : torch.Tensor
        Shape [N_modes p, N_modes p].
    H : torch.Tensor
        Shape [N_modes, N_modes p].
    """
    n, p = coef.shape
    kw = dict(device=coef.device, dtype=coef.dtype)
    A = torch.zeros(n * p, n * p, **kw)
    for j in range(p):
        A[:n, j * n:(j + 1) * n] = torch.diag(coef[:, j])
    if p > 1:
        A[n:, :-n] = torch.eye(n * (p - 1), **kw)
    H = torch.zeros(n, n * p, **kw)
    H[:, :n] = torch.eye(n, **kw)
    return A, H


def identify_ar_model(x, order, max_radius=None, R=None, R_fraction=0.1):
    """
    Identify per-mode AR(p) models from a window of pseudo-open-loop
    telemetry:

        1. c_ij by least squares (`estimate_ar`), roots shrunk to
           max_radius if given (`limit_ar_radius`)
        2. Q_ii = var(e_i), e the one-step prediction errors of the final
           coefficients (positive by construction)
        3. R = R_fraction * diag(Q), unless R is given
        4. companion form (`ar_companion`)

    Parameters
    ----------
    x : torch.Tensor
        Pseudo-open-loop telemetry, shape [N_frames, N_modes].
    order : int
        AR order p.
    max_radius : float or None
        Largest root modulus of each mode's AR polynomial.
    R : torch.Tensor or None
        Known noise covariance [N_modes, N_modes].
    R_fraction : float
        R_ii / Q_ii when R is None (> 0, the DARE needs R^{-1}).

    Returns
    -------
    IdentifiedModel
        A [n p, n p], H [n, n p], Q [n p, n p] (only the first block is
        non-zero), R [n, n], Q_unclipped = Q, A_ar1 = diag(c_1).
    """
    coef, e = estimate_ar(x, order)
    if max_radius is not None:
        coef = limit_ar_radius(coef, max_radius)
        e = ar_prediction_errors(x, coef)
    n = x.shape[1]
    q = e.var(dim=0)
    if R is None:
        if R_fraction <= 0:
            raise ValueError("R_fraction must be > 0")
        R = torch.diag(R_fraction * q)
    A, H = ar_companion(coef)
    Q = torch.zeros_like(A)
    Q[:n, :n] = torch.diag(q)
    return IdentifiedModel(A, H, Q, R, Q, A[:n, :n].clone())
