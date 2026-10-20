import numpy as np
import pytest
import torch

import kalman.Kalman_tools as kt
import kalman.Kalman_tools_numpy as kn

N_MODES = 4


@pytest.fixture
def x():
    """AR(1) telemetry with white measurement noise, float32, [N_frames, N_modes]."""
    rng = np.random.default_rng(0)
    a = np.linspace(0.99, 0.8, N_MODES)
    x = np.zeros((3000, N_MODES))
    for k in range(1, len(x)):
        x[k] = a * x[k - 1] + rng.standard_normal(N_MODES)
    x += 0.1 * rng.standard_normal(x.shape)
    return x.astype(np.float32)


@pytest.fixture
def R():
    return np.diag(np.full(N_MODES, 0.01, np.float32))


def t(a):
    return torch.from_numpy(a)


def test_atmospheric_state_matches_torch():
    rng = np.random.default_rng(1)
    z, u = rng.standard_normal((2, 10, N_MODES)).astype(np.float32)
    for delay in (0, 1, 2, 3):
        x = kn.estimate_atmospheric_state(z, u, delay)
        np.testing.assert_allclose(x, kt.estimate_atmospheric_state(t(z), t(u), delay).numpy())
    np.testing.assert_array_equal(kn.estimate_atmospheric_state(z, u, 3)[0], z[3] + u[0])


@pytest.mark.parametrize("method", ["least_squares", "diagonal", "instrumental_variable"])
def test_estimate_A_matches_torch(x, method):
    A = kn.estimate_A(x, method)
    assert A.dtype == np.float32
    np.testing.assert_allclose(A, kt.estimate_A(t(x), method).numpy(), atol=1e-5)


def test_estimate_A_unknown_method(x):
    with pytest.raises(ValueError):
        kn.estimate_A(x, "foo")


def test_estimate_raw_Q_matches_torch_and_keeps_dtype(x):
    A = kn.estimate_A(x)
    Q_raw = kn.estimate_raw_Q(x, A)
    assert Q_raw.dtype == np.float32
    np.testing.assert_allclose(Q_raw, kt.estimate_raw_Q(t(x), t(A)).numpy(), rtol=1e-5)


@pytest.mark.parametrize("force_positive", [False, True])
def test_estimate_Q_matches_torch(x, R, force_positive):
    A = kn.estimate_A(x)
    # Large R so that the subtraction produces negative eigenvalues
    R_big = 50 * R
    Q_raw = kn.estimate_raw_Q(x, A)
    Q = kn.estimate_Q(Q_raw, A, R_big, force_positive)
    Q_t = kt.estimate_Q(t(Q_raw), t(A), t(R_big), force_positive).numpy()
    np.testing.assert_allclose(Q, Q_t, atol=1e-5)
    eigenvalues = np.linalg.eigvalsh(Q)
    if force_positive:
        assert eigenvalues.min() > -1e-6
    else:
        assert eigenvalues.min() < 0


def test_estimate_R_matches_torch():
    rng = np.random.default_rng(2)
    recon = rng.standard_normal((20, N_MODES)).astype(np.float32)
    ref = rng.random(20).astype(np.float32)
    R = kn.estimate_R(recon, ref, 1e4, 2.0)
    np.testing.assert_allclose(R, kt.estimate_R(t(recon), t(ref), 1e4, 2.0).numpy(), rtol=1e-5)


def run_filters(x, R, measurement_shape):
    A = kn.estimate_A(x)
    Q = kn.estimate_raw_Q(x, A)
    H = np.eye(N_MODES, dtype=np.float32)
    P0 = np.eye(N_MODES, dtype=np.float32)
    m_np = kn.KalmanMatrices(A, H, Q, R)
    m_t = kt.KalmanMatrices(t(A), t(H), t(Q), t(R))
    s_np = kn.KalmanState(x[0], P0)
    s_t = kt.KalmanState(t(x[0]), t(P0))
    for k in range(1, 50):
        s_np, K_np = kn.kalman_filter(x[k].reshape(measurement_shape), s_np, m_np)
        s_t, K_t = kt.kalman_filter(t(x[k]), s_t, m_t)
    return s_np, K_np, s_t, K_t


@pytest.mark.parametrize("measurement_shape", [(N_MODES,), (N_MODES, 1)])
def test_kalman_filter_matches_torch(x, R, measurement_shape):
    s_np, K_np, s_t, K_t = run_filters(x, R, measurement_shape)
    assert s_np.x.shape == (N_MODES,)
    assert s_np.x.dtype == s_np.P.dtype == np.float32
    np.testing.assert_allclose(s_np.x, s_t.x.numpy(), atol=1e-5)
    np.testing.assert_allclose(s_np.P, s_t.P.numpy(), atol=1e-6)
    np.testing.assert_allclose(K_np, K_t.numpy(), atol=1e-5)
