import pytest
import torch

from kalman.Kalman_tools import (
    KalmanMatrices,
    KalmanState,
    estimate_A,
    estimate_A_diagonal,
    estimate_atmospheric_state,
    estimate_Q,
    estimate_R,
    estimate_raw_Q,
    kalman_filter,
)

N_MODES = 4


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(0)


def simulate_ar1(a, q_std, n_frames, generator):
    """Simulate x[k+1] = diag(a) @ x[k] + w[k], w ~ N(0, q_std**2)."""
    x = torch.zeros(n_frames, a.numel(), dtype=torch.float64)
    for k in range(n_frames - 1):
        w = q_std * torch.randn(a.numel(), generator=generator, dtype=torch.float64)
        x[k + 1] = a * x[k] + w
    return x


def test_atmospheric_state_no_delay():
    z = torch.arange(6.0).reshape(3, 2)
    u = torch.ones(3, 2)
    assert torch.equal(estimate_atmospheric_state(z, u, 0), z + u)


def test_atmospheric_state_with_delay():
    z = torch.arange(10.0).reshape(5, 2)
    u = 100 * torch.arange(10.0).reshape(5, 2)
    x = estimate_atmospheric_state(z, u, 2)
    assert x.shape == (3, 2)
    assert torch.equal(x, z[2:] + u[:-2])


def test_atmospheric_state_shape_mismatch():
    with pytest.raises(ValueError):
        estimate_atmospheric_state(torch.zeros(3, 2), torch.zeros(3, 3), 0)


def test_estimate_A_recovers_ar1(generator):
    a = torch.tensor([0.99, 0.9, 0.7, 0.5], dtype=torch.float64)
    x = simulate_ar1(a, 0.1, 20000, generator)

    A = estimate_A(x)
    A_diag = estimate_A_diagonal(x)

    assert A.shape == (N_MODES, N_MODES)
    torch.testing.assert_close(A, torch.diag(a), atol=0.02, rtol=0)
    torch.testing.assert_close(A_diag, torch.diag(a), atol=0.02, rtol=0)


def test_estimate_Q_removes_measurement_noise(generator):
    a = torch.tensor([0.99, 0.9, 0.7, 0.5], dtype=torch.float64)
    q_std, r_std = 0.1, 0.05
    x_true = simulate_ar1(a, q_std, 50000, generator)
    x_meas = x_true + r_std * torch.randn(
        x_true.shape, generator=generator, dtype=torch.float64
    )

    A = torch.diag(a)
    R = r_std**2 * torch.eye(N_MODES, dtype=torch.float64)
    Q = estimate_Q(estimate_raw_Q(x_meas, A), A, R)

    torch.testing.assert_close(Q, Q.T)
    torch.testing.assert_close(
        torch.diag(Q), torch.full((N_MODES,), q_std**2, dtype=torch.float64),
        atol=1e-3, rtol=0,
    )


def test_estimate_R_symmetric_psd(generator):
    n_pix = 20
    recon = torch.randn(n_pix, N_MODES, generator=generator, dtype=torch.float64)
    ref_intensity = torch.rand(n_pix, generator=generator, dtype=torch.float64)

    R = estimate_R(recon, ref_intensity, nPhotons=1e4, ron=2.0)

    assert R.shape == (N_MODES, N_MODES)
    torch.testing.assert_close(R, R.T)
    assert torch.linalg.eigvalsh(R).min() > 0


def test_kalman_filter_matches_textbook_update(generator):
    def spd(n):
        M = torch.randn(n, n, generator=generator, dtype=torch.float64)
        return M @ M.T + n * torch.eye(n, dtype=torch.float64)

    matrices = KalmanMatrices(
        A=0.9 * torch.eye(N_MODES, dtype=torch.float64),
        H=torch.eye(N_MODES, dtype=torch.float64),
        Q=spd(N_MODES),
        R=spd(N_MODES),
    )
    state = KalmanState(
        x=torch.randn(N_MODES, generator=generator, dtype=torch.float64),
        P=spd(N_MODES),
    )
    y = torch.randn(N_MODES, generator=generator, dtype=torch.float64)

    new_state, K = kalman_filter(y, state, matrices)

    A, H, Q, R = matrices.A, matrices.H, matrices.Q, matrices.R
    x_pred = A @ state.x
    P_pred = A @ state.P @ A.T + Q
    K_ref = P_pred @ H.T @ torch.linalg.inv(H @ P_pred @ H.T + R)
    x_ref = x_pred + K_ref @ (y - H @ x_pred)
    P_ref = (torch.eye(N_MODES, dtype=torch.float64) - K_ref @ H) @ P_pred

    torch.testing.assert_close(K, K_ref)
    torch.testing.assert_close(new_state.x, x_ref)
    torch.testing.assert_close(new_state.P, P_ref)
    assert new_state.P.dtype == torch.float64
