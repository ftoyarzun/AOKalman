import torch
from dataclasses import dataclass

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


def estimate_A(x):
    """
    Estimate the state-transition matrix A from:

        x[k+1] = A @ x[k] + w[k]

    using least squares.

    Parameters
    ----------
    x : torch.Tensor
        Estimated atmospheric states, shape [N_frames, N_modes].

    Returns
    -------
    A : torch.Tensor
        State-transition matrix, shape [N_modes, N_modes].
    """
    X = x[:-1]   # x_k
    Xp = x[1:]   # x_{k+1}

    # Solve X @ A.T ~= Xp
    A = torch.linalg.lstsq(X, Xp).solution.T

    return A


def estimate_A_diagonal(x):
    """
    Estimate a diagonal AR(1) state-transition matrix:

        x_i[k+1] = a_i * x_i[k] + w_i[k]
    """
    X = x[:-1]
    Xp = x[1:]

    a = torch.sum(Xp * X, dim=0) / torch.sum(X * X, dim=0)

    return torch.diag(a)


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

def estimate_Q(Q_raw, A, R):
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

    Returns
    -------
    Q : torch.Tensor
        Estimated process covariance.
    """
    Q = Q_raw - R - A @ R @ A.T

    # Numerical symmetrization
    Q = 0.5 * (Q + Q.T)

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