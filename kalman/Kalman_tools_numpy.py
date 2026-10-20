import numpy as np

from dataclasses import dataclass

@dataclass
class KalmanData:
    wfs_measurements: np.ndarray
    modal_dm_commands: np.ndarray

@dataclass
class KalmanState:
    x: np.ndarray
    P: np.ndarray


@dataclass
class KalmanMatrices:
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

def estimate_atmospheric_state(z, u, delay):
    """
    Estimate the atmospheric modal state from closed-loop telemetry.

    Parameters
    ----------
    z : np.ndarray
        WFS modal measurements, shape [N_frames, N_modes].
    u : np.ndarray
        Modal DM commands, shape [N_frames, N_modes].
    delay : int
        Frames between computing a command and seeing it in the WFS
        measurement: z_k contains u_{k-delay}.

    Returns
    -------
    x : np.ndarray
        Estimated atmospheric state, x_k ~= z_k + u_{k-delay}.
        Shape [N_frames - delay, N_modes].
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
    x : np.ndarray
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
    A : np.ndarray
        State-transition matrix, shape [N_modes, N_modes].
    """
    Xm = x[:-2]   # x_{k-1}
    X = x[1:-1]   # x_k
    Xp = x[2:]    # x_{k+1}

    if method == "least_squares":
        # Solve X @ A.T ~= Xp
        return np.linalg.lstsq(X, Xp, rcond=None)[0].T

    if method == "diagonal":
        a = np.sum(Xp * X, axis=0) / np.sum(X * X, axis=0)
        return np.diag(a)

    if method == "instrumental_variable":
        # A @ (X.T @ Xm) = Xp.T @ Xm  <=>  (Xm.T @ X) @ A.T = Xm.T @ Xp
        return np.linalg.solve(Xm.T @ X, Xm.T @ Xp).T

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
    x : np.ndarray
        Estimated atmospheric states, shape [N_frames, N_modes].

    A : np.ndarray
        State-transition matrix, shape [N_modes, N_modes].

    Returns
    -------
    Q_raw : np.ndarray
        Raw prediction-error covariance, shape [N_modes, N_modes],
        in the dtype of x.
    """
    X = x[:-1]
    Xp = x[1:]

    prediction_error = Xp - X @ A.T

    # np.cov always returns float64
    return np.cov(prediction_error.T).astype(x.dtype, copy=False)

def estimate_Q(Q_raw, A, R, force_positive=False):
    """
    Estimate the atmospheric process covariance Q by removing
    the contribution from WFS measurement noise.

        Q ~= Q_raw - R - A @ R @ A.T

    Parameters
    ----------
    Q_raw : np.ndarray
        Raw prediction-error covariance.
    A : np.ndarray
        State-transition matrix.
    R : np.ndarray
        WFS measurement-noise covariance.
    force_positive : bool
        A covariance cannot have negative eigenvalues, but the subtraction
        above can produce them when R is large compared to Q_raw (low SNR)
        or A is poorly estimated; the Kalman filter can then diverge.
        If True, negative eigenvalues are set to zero (nearest positive
        semi-definite matrix).

    Returns
    -------
    Q : np.ndarray
        Estimated process covariance.
    """
    Q = Q_raw - R - A @ R @ A.T

    # Numerical symmetrization
    Q = 0.5 * (Q + Q.T)

    if force_positive:
        eigenvalues, eigenvectors = np.linalg.eigh(Q)
        Q = eigenvectors @ np.diag(np.clip(eigenvalues, 0, None)) @ eigenvectors.T

    return Q


def estimate_R(recon, ref_intensity, nPhotons, ron):

    cov_ron = ron**2/nPhotons**2 * recon.T @ recon
    cov_photon = 1/nPhotons * (ref_intensity.reshape(-1,1) * recon).T @ recon

    return (cov_ron + cov_photon).squeeze()


def kalman_filter(
    measurement: np.ndarray,
    state: KalmanState,
    matrices: KalmanMatrices,
) -> tuple[KalmanState, np.ndarray]:
    """
    Perform one Kalman filter iteration.

    Parameters
    ----------
    measurement : np.ndarray
        Current measurement y_k, shape [N_meas] or [N_meas, 1]
        (flattened, so a column read from shared memory is fine).

    state : KalmanState
        Previous posterior state estimate:
            state.x = x_{k-1|k-1}, shape [N_state]
            state.P = P_{k-1|k-1}, shape [N_state, N_state]

    matrices : KalmanMatrices
        Kalman filter matrices.

    Returns
    -------
    state : KalmanState
        Updated posterior state:
            state.x = x_{k|k}
            state.P = P_{k|k}

    K : np.ndarray
        Kalman gain.
    """

    A = matrices.A
    H = matrices.H
    Q = matrices.Q
    R = matrices.R

    # A [N_meas, 1] measurement would broadcast the innovation to a matrix
    measurement = np.ravel(measurement)

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
    K = np.linalg.solve(
        S,
        H @ P_pred,
    ).T

    x_new = x_pred + K @ innovation

    # Joseph form for numerical stability
    I = np.eye(
        P_pred.shape[0],
        dtype=P_pred.dtype,
    )

    IKH = I - K @ H

    P_new = (
        IKH @ P_pred @ IKH.T
        + K @ R @ K.T
    )

    return KalmanState(x=x_new, P=P_new), K
