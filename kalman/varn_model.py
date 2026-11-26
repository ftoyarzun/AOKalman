"""VARN-LQG -- the model: a VAR(N) model of all the modes, its state-space form and its Kalman
predictor. Pure numpy, no side effects: used by optimizer_varn_lqg.py, and importable from a
notebook.

  fit_var      VAR(N) by least squares, and Q, the covariance of its prediction residuals
  companion    the state-space form A of the VAR(N) model
  kalman_gain  the steady-state Kalman gain (Riccati equation, iterated)
  ahead        A^delay, fractional delays included
  split_delay  delay = n + f
"""
import numpy as np


def split_delay(delay):
    """The loop delay, frames, may be fractional (da.delay_exact, e.g. 3.3): n whole frames and f,
    the command reaching the WFS being (1 - f) output(k - n) + f output(k - n - 1)."""
    delay = max(1.0, float(delay))
    n = int(delay + 1e-9)
    f = delay - n
    return n, (f if f > 1e-9 else 0.0)


def ahead(A, delay):
    """The state `delay` frames ahead, may be fractional: (1 - f) A^n + f A^(n+1), so that the
    blend of two outputs the DM applies predicts the right frame on average."""
    n, f = split_delay(delay)
    An = np.linalg.matrix_power(A, n)
    return (1.0 - f) * An + f * (An @ A) if f else An


def fit_var(pol, order):
    """
    VAR(order) model of pol by least squares, and the covariance of its prediction residuals.

        pol_t+1 = var @ [pol_t, pol_t-1, ..., pol_t-order+1] + e_t+1,   Q = cov(e)

    Parameters
    ----------
    pol : ndarray, shape [n_modes, frames]
        The pseudo open loop (the turbulence). Its mean is removed first.
    order : int
        N, at least 1.

    Returns
    -------
    var : ndarray, shape [n_modes, order * n_modes]
    Q : ndarray, shape [n_modes, n_modes]
        Covariance of the one-step prediction residuals e.
    """
    n, frames = pol.shape
    pol = pol - pol.mean(axis=1, keepdims=True)
    # row block j: pol_t-j, for t = order-1 ... frames-2
    past = np.vstack([pol[:, order - 1 - j:frames - 1 - j] for j in range(order)])
    future = pol[:, order:]                                  # pol_t+1
    # min-norm least squares: modes that are exactly 0 (the CNN's 80-95) get zero coefficients
    var = np.linalg.lstsq(past.T, future.T, rcond=None)[0].T
    residual = future - var @ past
    return var, np.atleast_2d(np.cov(residual))


def companion(var):
    """The state-space form of the VAR(N) model, state [pol_t, ..., pol_t-N+1]:
    A = [[var], [I 0]] (N n x N n): the newest block from the model, the others shifted."""
    n = var.shape[0]
    A = np.zeros((var.shape[1], var.shape[1]))
    A[:n] = var
    A[n:, :-n] = np.eye(var.shape[1] - n)
    return A


def kalman_gain(A, Q, R, n_iter, tol=1e-10):
    """
    Steady-state Kalman gain for H = [I 0 ... 0] (the newest block is measured), from the
    Riccati equation iterated from P = Q until P changes by less than tol (relative):

        K = P H^T (H P H^T + R)^-1,   P <- A (P - K H P) A^T + Q

    Returns K, shape [N n, n], and the number of iterations.
    """
    n = R.shape[0]
    P = Q.copy()
    for it in range(1, n_iter + 1):
        K = P[:, :n] @ np.linalg.inv(P[:n, :n] + R)       # P H^T = P[:, :n], H P H^T = P[:n, :n]
        P_new = A @ (P - K @ P[:n]) @ A.T + Q
        converged = np.max(np.abs(P_new - P)) <= tol * np.max(np.abs(P_new))
        P = P_new
        if converged:
            break
    return P[:, :n] @ np.linalg.inv(P[:n, :n] + R), it
