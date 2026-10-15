import pytest
import torch

from kalman.Kalman_tools import (
    KalmanMatrices,
    KalmanState,
    RecursiveCovariance,
    apply_structure,
    ar2_noise_variance,
    ar2_state,
    augment_ar2,
    autocovariances,
    build_realtime_matrix,
    estimate_ar2,
    estimate_ar2_from_autocovariances,
    estimate_Q_ar2,
    identify_model,
    estimate_A,
    estimate_A_from_covariances,
    estimate_Q_from_covariances,
    estimate_R_from_covariances,
    estimate_R_from_telemetry,
    lagged_covariances,
    validate_model,
    zernike_radial_groups,
    estimate_atmospheric_state,
    estimate_Q,
    estimate_R,
    estimate_raw_Q,
    innovation_whitener,
    kalman_filter,
    limit_spectral_radius,
    realtime_step,
    steady_state_gain,
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


@pytest.mark.parametrize(
    "method", ["least_squares", "diagonal", "instrumental_variable"]
)
def test_estimate_A_recovers_ar1(generator, method):
    a = torch.tensor([0.99, 0.9, 0.7, 0.5], dtype=torch.float64)
    x = simulate_ar1(a, 0.1, 20000, generator)

    A = estimate_A(x, method=method)

    assert A.shape == (N_MODES, N_MODES)
    torch.testing.assert_close(A, torch.diag(a), atol=0.03, rtol=0)


def test_estimate_A_default_is_least_squares(generator):
    x = torch.randn(100, N_MODES, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(estimate_A(x), estimate_A(x, method="least_squares"))


def test_estimate_A_instrumental_variable_unbiased_by_noise(generator):
    a = torch.tensor([0.95, 0.9, 0.8, 0.7], dtype=torch.float64)
    x_true = simulate_ar1(a, 0.1, 200000, generator)
    # measurement noise comparable to the signal
    x_meas = x_true + 0.3 * torch.randn(
        x_true.shape, generator=generator, dtype=torch.float64
    )

    a_ls = torch.diag(estimate_A(x_meas, method="least_squares"))
    a_iv = torch.diag(estimate_A(x_meas, method="instrumental_variable"))

    assert torch.all(a_ls < a - 0.1)  # least squares is pulled towards zero
    torch.testing.assert_close(a_iv, a, atol=0.03, rtol=0)


def test_estimate_A_unknown_method():
    with pytest.raises(ValueError):
        estimate_A(torch.zeros(10, N_MODES), method="kalman")


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


def test_estimate_Q_force_positive():
    A = torch.zeros(2, 2, dtype=torch.float64)
    R = torch.eye(2, dtype=torch.float64)
    # R larger than Q_raw in one direction -> negative eigenvalue
    Q_raw = torch.diag(torch.tensor([3.0, 0.5], dtype=torch.float64))

    Q = estimate_Q(Q_raw, A, R)
    Q_pos = estimate_Q(Q_raw, A, R, force_positive=True)

    assert torch.linalg.eigvalsh(Q).min() < 0
    assert torch.linalg.eigvalsh(Q_pos).min() >= 0
    torch.testing.assert_close(Q_pos, torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64)))


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


# -------------------------------------------------------------
# Steady-state gain and real-time step
# -------------------------------------------------------------

def random_model(generator, n=N_MODES, rho=0.95, scale=1.0):
    """Stable full A (spectral radius rho), SPD Q and R, H = I, all float64."""
    M = torch.randn(n, n, generator=generator, dtype=torch.float64)
    A = rho * M / torch.linalg.eigvals(M).abs().max()
    B = torch.randn(n, n, generator=generator, dtype=torch.float64)
    Q = scale * (0.1 * B @ B.T + 0.05 * torch.eye(n, dtype=torch.float64))
    C = torch.randn(n, n, generator=generator, dtype=torch.float64)
    R = scale * (0.1 * C @ C.T + 0.1 * torch.eye(n, dtype=torch.float64))
    H = torch.eye(n, dtype=torch.float64)
    return A, H, Q, R


def kalman_filter_limit(A, H, Q, R, n_iter=3000):
    state = KalmanState(x=torch.zeros(A.shape[0], dtype=A.dtype), P=Q.clone())
    matrices = KalmanMatrices(A=A, H=H, Q=Q, R=R)
    y = torch.zeros(H.shape[0], dtype=A.dtype)
    for _ in range(n_iter):
        state, K = kalman_filter(y, state, matrices)
    return K, A @ state.P @ A.T + Q


@pytest.mark.parametrize("method", ["doubling", "riccati"])
def test_steady_state_gain_is_kalman_filter_limit(generator, method):
    A, H, Q, R = random_model(generator)
    K_ref, P_ref = kalman_filter_limit(A, H, Q, R)

    res = steady_state_gain(A, H, Q, R, tol=1e-12, method=method,
                            compute_dtype=torch.float64)

    assert res.converged
    torch.testing.assert_close(res.K, K_ref, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(res.P, P_ref, atol=1e-9, rtol=1e-7)
    assert res.residual < 1e-10


def test_steady_state_gain_float32_internal_returns_input_dtype(generator):
    A, H, Q, R = random_model(generator, rho=0.999)
    K_ref, _ = kalman_filter_limit(A, H, Q, R, n_iter=20000)

    res = steady_state_gain(A, H, Q, R, tol=1e-5)

    assert res.converged
    assert res.K.dtype == torch.float64 and res.K.device == A.device
    torch.testing.assert_close(res.K, K_ref, atol=1e-4, rtol=1e-3)


def test_steady_state_gain_opd_units_in_float32(generator):
    # OPD-like variances (~1e-16 m^2) would underflow products in float32
    # without the internal rescaling.
    A, H, Q, R = random_model(generator)
    K_ref = steady_state_gain(A, H, Q, R, compute_dtype=torch.float64).K

    s = 1e-16
    res = steady_state_gain(A.float(), H.float(), s * Q.float(), s * R.float(), tol=1e-5)

    assert res.converged and res.K.dtype == torch.float32
    torch.testing.assert_close(res.K.double(), K_ref, atol=1e-4, rtol=1e-3)
    assert torch.isfinite(res.P).all() and res.P.abs().max() < 1e-13


def test_steady_state_gain_non_convergence(generator):
    A, H, Q, R = random_model(generator, rho=0.999)
    with pytest.warns(RuntimeWarning):
        res = steady_state_gain(A, H, Q, R, method="riccati", max_iter=2)
    assert not res.converged
    with pytest.raises(RuntimeError):
        steady_state_gain(A, H, Q, R, method="riccati", max_iter=2, on_fail="raise")


def test_innovation_whitener_gives_identity_covariance(generator):
    A, H, Q, R = random_model(generator)
    P = steady_state_gain(A, H, Q, R, compute_dtype=torch.float64).P
    W = innovation_whitener(H, P, R)
    S = H @ P @ H.T + R
    torch.testing.assert_close(W @ S @ W.T, torch.eye(N_MODES, dtype=torch.float64))


@pytest.mark.parametrize("delay", [1, 2])
def test_realtime_step_matches_converged_kalman_filter(generator, delay):
    A, H, Q, R = random_model(generator)
    K_ref, P_pred = kalman_filter_limit(A, H, Q, R)
    matrices = KalmanMatrices(A=A, H=H, Q=Q, R=R)
    # posterior covariance that makes the next kalman_filter call use P_pred
    P_post = torch.linalg.solve(A, torch.linalg.solve(A, (P_pred - Q).T).T)

    res = steady_state_gain(A, H, Q, R, tol=1e-12, compute_dtype=torch.float64)
    W = innovation_whitener(H, res.P, R)
    M = build_realtime_matrix(A, H, res.K, delay, whiten=W)
    assert M.shape == (3 * N_MODES, 2 * N_MODES)

    x = torch.randn(N_MODES, generator=generator, dtype=torch.float64)
    for _ in range(5):
        y = torch.randn(N_MODES, generator=generator, dtype=torch.float64)
        new_state, _ = kalman_filter(y, KalmanState(x=x, P=P_post), matrices)
        u_ref = torch.linalg.matrix_power(A, delay) @ new_state.x
        e_ref = W @ (y - H @ A @ x)

        x_rt, out = realtime_step(M, x, y)

        torch.testing.assert_close(x_rt, new_state.x)
        torch.testing.assert_close(out[:N_MODES], u_ref)
        torch.testing.assert_close(out[N_MODES:], e_ref)
        _, u_only = realtime_step(M, x, y, n_command=N_MODES)
        torch.testing.assert_close(u_only, u_ref)
        x = x_rt


def test_realtime_step_batched(generator):
    A, H, Q, R = random_model(generator)
    K = steady_state_gain(A, H, Q, R, compute_dtype=torch.float64).K
    M = build_realtime_matrix(A, H, K, 2)
    x = torch.randn(3, N_MODES, generator=generator, dtype=torch.float64)
    y = torch.randn(3, N_MODES, generator=generator, dtype=torch.float64)
    xb, ub = realtime_step(M, x, y)
    for b in range(3):
        xs, us = realtime_step(M, x[b], y[b])
        torch.testing.assert_close(xb[b], xs)
        torch.testing.assert_close(ub[b], us)


# -------------------------------------------------------------
# Identification from lagged covariances
# -------------------------------------------------------------

def simulate_linear(A, Q_chol, n_frames, generator):
    """Simulate x[k+1] = A x[k] + Q_chol @ w[k], w ~ N(0, I), float64."""
    n = A.shape[0]
    x = torch.zeros(n_frames, n, dtype=torch.float64)
    w = torch.randn(n_frames, n, generator=generator, dtype=torch.float64) @ Q_chol.T
    for k in range(n_frames - 1):
        x[k + 1] = A @ x[k] + w[k]
    return x


def coupled_A():
    """Full A with off-diagonal coupling and spectral radius < 1."""
    return torch.tensor(
        [[0.95, 0.10, 0.00, 0.00],
         [-0.10, 0.93, 0.05, 0.00],
         [0.00, 0.00, 0.90, 0.20],
         [0.00, 0.00, -0.15, 0.85]],
        dtype=torch.float64,
    )


def test_lagged_covariances_definition(generator):
    x = torch.randn(50, 3, generator=generator, dtype=torch.float64)
    C = lagged_covariances(x, max_lag=2)
    assert len(C) == 3
    for l in range(3):
        ref = sum(torch.outer(x[k + l], x[k]) for k in range(50 - l)) / (50 - l)
        torch.testing.assert_close(C[l], ref)


def test_recursive_covariance_block_equals_single_updates(generator):
    x = torch.randn(40, 3, generator=generator, dtype=torch.float64)
    rc1 = RecursiveCovariance(3, 0.9, max_lag=2, dtype=torch.float64)
    rc2 = RecursiveCovariance(3, 0.9, max_lag=2, dtype=torch.float64)
    for k in range(40):
        rc1.update(x[k])
    rc2.update(x[:7])
    rc2.update(x[7:8])
    rc2.update(x[8:])
    for a, b in zip(rc1.covariances(), rc2.covariances()):
        torch.testing.assert_close(a, b)


def test_recursive_covariance_matches_weighted_definition(generator):
    lam, N = 0.95, 200
    x = torch.randn(N, 3, generator=generator, dtype=torch.float64)
    rc = RecursiveCovariance(3, lam, max_lag=1, dtype=torch.float64)
    rc.update(x)
    C0, C1 = rc.covariances()
    w = (1 - lam) * lam ** torch.arange(N - 1, -1, -1, dtype=torch.float64)
    C0_ref = (x * w[:, None]).T @ x / w.sum()
    C1_ref = (x[1:] * w[1:, None]).T @ x[:-1] / w.sum()
    torch.testing.assert_close(C0, C0_ref)
    torch.testing.assert_close(C1, C1_ref)


def test_recursive_covariance_tracks_change(generator):
    # the AR coefficient changes half-way; the forgotten estimate follows it
    a1 = torch.full((N_MODES,), 0.5, dtype=torch.float64)
    a2 = torch.full((N_MODES,), 0.95, dtype=torch.float64)
    x = torch.cat([simulate_ar1(a1, 0.1, 20000, generator),
                   simulate_ar1(a2, 0.1, 20000, generator)])
    rc = RecursiveCovariance(N_MODES, 1 - 1 / 2000, max_lag=1, dtype=torch.float64)
    rc.update(x)
    A = estimate_A_from_covariances(*rc.covariances(), structure="diagonal")
    torch.testing.assert_close(torch.diagonal(A), a2, atol=0.05, rtol=0)


def test_estimate_A_from_covariances_matches_estimate_A(generator):
    a = torch.tensor([0.99, 0.9, 0.7, 0.5], dtype=torch.float64)
    x = simulate_ar1(a, 0.1, 20000, generator)
    C0, C1, C2 = lagged_covariances(x)
    for method in ["least_squares", "diagonal", "instrumental_variable"]:
        torch.testing.assert_close(
            estimate_A_from_covariances(C0, C1, C2, method=method),
            estimate_A(x, method=method),
            atol=5e-3, rtol=0,
        )


def test_estimate_A_from_covariances_iv_full_unbiased_by_noise(generator):
    A_true = coupled_A()
    x = simulate_linear(A_true, 0.1 * torch.eye(N_MODES, dtype=torch.float64),
                        200000, generator)
    x = x + 0.2 * torch.randn(x.shape, generator=generator, dtype=torch.float64)
    C0, C1, C2 = lagged_covariances(x)

    A_ls = estimate_A_from_covariances(C0, C1, C2, method="least_squares")
    A_iv = estimate_A_from_covariances(C0, C1, C2, method="instrumental_variable")

    assert torch.linalg.norm(A_ls - A_true) > 0.1
    torch.testing.assert_close(A_iv, A_true, atol=0.03, rtol=0)


def test_estimate_A_from_covariances_block_and_ridge(generator):
    A_true = coupled_A()
    A_true[1, 2] = 0.0     # block diagonal with blocks {0, 1} and {2, 3}
    x = simulate_linear(A_true, 0.1 * torch.eye(N_MODES, dtype=torch.float64),
                        50000, generator)
    C0, C1, C2 = lagged_covariances(x)
    groups = torch.tensor([0, 0, 1, 1])

    A_blk = estimate_A_from_covariances(C0, C1, structure="block", groups=groups)
    assert torch.all(A_blk[:2, 2:] == 0) and torch.all(A_blk[2:, :2] == 0)
    torch.testing.assert_close(A_blk, A_true, atol=0.02, rtol=0)

    A_ridge = estimate_A_from_covariances(C0, C1, ridge=1.0)
    assert torch.linalg.norm(A_ridge) < torch.linalg.norm(A_blk)

    with pytest.raises(ValueError):
        estimate_A_from_covariances(C0, C1, structure="block")


def test_zernike_radial_groups():
    # Noll 2..11: tip, tilt (n=1), defocus, astig x2 (n=2), coma x2, trefoil x2 (n=3), spherical (n=4)
    torch.testing.assert_close(
        zernike_radial_groups(10),
        torch.tensor([1, 1, 2, 2, 2, 3, 3, 3, 3, 4]),
    )


@pytest.mark.parametrize("structure", ["full", "diagonal"])
def test_estimate_R_from_telemetry_recovers_noise(generator, structure):
    A_true = coupled_A()
    x = simulate_linear(A_true, 0.1 * torch.eye(N_MODES, dtype=torch.float64),
                        200000, generator)
    r_std = torch.tensor([0.05, 0.1, 0.15, 0.2], dtype=torch.float64)
    x = x + r_std * torch.randn(x.shape, generator=generator, dtype=torch.float64)

    R = estimate_R_from_telemetry(x, A_true, structure=structure)

    torch.testing.assert_close(R, R.T)
    atol = 3e-3 if structure == "full" else 1e-3
    torch.testing.assert_close(R, torch.diag(r_std**2), atol=atol, rtol=0)


def test_estimate_R_floor_keeps_R_positive_definite():
    C0 = torch.eye(2, dtype=torch.float64)
    A = 0.5 * torch.eye(2, dtype=torch.float64)
    C1 = 0.6 * torch.eye(2, dtype=torch.float64)   # implies negative noise
    for structure in ["full", "diagonal"]:
        R = estimate_R_from_covariances(C0, C1, A, structure=structure)
        assert torch.linalg.eigvalsh(R).min() > 0


def test_estimate_Q_from_covariances_matches_window_estimate(generator):
    a = torch.tensor([0.99, 0.9, 0.7, 0.5], dtype=torch.float64)
    q_std, r_std = 0.1, 0.05
    x = simulate_ar1(a, q_std, 50000, generator)
    x = x + r_std * torch.randn(x.shape, generator=generator, dtype=torch.float64)
    A = torch.diag(a)
    R = r_std**2 * torch.eye(N_MODES, dtype=torch.float64)
    C0, C1 = lagged_covariances(x, max_lag=1)

    Q = estimate_Q_from_covariances(C0, C1, A, R)

    torch.testing.assert_close(Q, estimate_Q(estimate_raw_Q(x, A), A, R, force_positive=True),
                               atol=2e-4, rtol=0)
    torch.testing.assert_close(torch.diag(Q), torch.full((N_MODES,), q_std**2, dtype=torch.float64),
                               atol=1e-3, rtol=0)


def test_validate_model(generator):
    A, H, Q, R = random_model(generator)
    dare = steady_state_gain(A, H, Q, R, compute_dtype=torch.float64)
    check = validate_model(A, Q, R, dare)
    assert check.ok and check.reasons == () and check.spectral_radius < 1

    assert not validate_model(1.01 * A / 0.95, Q, R).ok
    Q_bad = Q - 2 * torch.diag(torch.diagonal(Q))
    bad = validate_model(A, Q_bad, R)
    assert not bad.ok and bad.negative_Q_fraction > 0.2
    assert not validate_model(A, Q, -R).ok
    with pytest.warns(RuntimeWarning):
        dare_bad = steady_state_gain(A, H, Q, R, method="riccati", max_iter=1)
    assert not validate_model(A, Q, R, dare_bad).ok
    A_nan = A.clone()
    A_nan[0, 0] = float("nan")
    assert not validate_model(A_nan, Q, R).ok


# -------------------------------------------------------------
# AR(2) augmentation
# -------------------------------------------------------------

def resonance(f, fs, rho):
    """AR(2) coefficients of a resonance at f (Hz), sampled at fs, pole radius rho."""
    w = 2 * torch.pi * torch.as_tensor(f, dtype=torch.float64) / fs
    return 2 * rho * torch.cos(w), torch.full_like(w, -rho**2)


def simulate_ar2(a1, a2, q_std, n_frames, generator):
    """x[k+1] = a1 x[k] + a2 x[k-1] + w[k] per mode (a randomly-excited damped sinusoid)."""
    x = torch.zeros(n_frames, a1.numel(), dtype=torch.float64)
    w = q_std * torch.randn(n_frames, a1.numel(), generator=generator, dtype=torch.float64)
    for k in range(1, n_frames - 1):
        x[k + 1] = a1 * x[k] + a2 * x[k - 1] + w[k]
    return x


def test_estimate_ar2_deterministic_damped_sinusoid():
    fs, f, rho = 1000.0, 40.0, 0.999
    k = torch.arange(4000, dtype=torch.float64)
    x = (rho**k * torch.sin(2 * torch.pi * f / fs * k + 0.3)).reshape(-1, 1)
    a1_true, a2_true = resonance([f], fs, rho)

    a1, a2 = estimate_ar2(x, [0], method="least_squares")

    torch.testing.assert_close(a1, a1_true, atol=5e-3, rtol=0)
    torch.testing.assert_close(a2, a2_true, atol=5e-3, rtol=0)


def test_estimate_ar2_noisy_vibration_iv_unbiased(generator):
    a1_true, a2_true = resonance([17.0, 43.0], 1000.0, 0.995)
    x = simulate_ar2(a1_true, a2_true, 0.1, 100000, generator)
    y = x + 0.5 * x.std() * torch.randn(x.shape, generator=generator, dtype=torch.float64)

    a1_ls, a2_ls = estimate_ar2(y, [0, 1], method="least_squares")
    a1_iv, a2_iv = estimate_ar2(y, [0, 1], method="instrumental_variable")

    assert torch.all((a2_ls - a2_true).abs() > 0.05)   # noise biases LS
    torch.testing.assert_close(a1_iv, a1_true, atol=5e-3, rtol=0)
    torch.testing.assert_close(a2_iv, a2_true, atol=5e-3, rtol=0)

    gamma = autocovariances(y, max_lag=20)
    r = ar2_noise_variance(gamma, a1_iv, a2_iv)
    r_true = (0.5 * x.std()) ** 2
    torch.testing.assert_close(r, torch.full((2,), float(r_true), dtype=torch.float64),
                               atol=0.1 * float(r_true), rtol=0)

    with pytest.raises(ValueError):
        estimate_ar2_from_autocovariances(gamma[:4], method="instrumental_variable")


def test_augment_ar2_structure_any_modes():
    n = 5
    A = 0.5 * torch.eye(n, dtype=torch.float64)
    A[0, 3] = 0.1      # mode 0 depends on mode 3: column 3 must be kept
    A[3, 4] = 0.2      # row 3 is replaced by its AR(2) model
    H = torch.eye(n, dtype=torch.float64)
    modes = [1, 3]
    a1 = torch.tensor([1.9, 1.8], dtype=torch.float64)
    a2 = torch.tensor([-0.95, -0.9], dtype=torch.float64)

    A_aug, H_aug = augment_ar2(A, H, modes, a1, a2)

    assert A_aug.shape == (7, 7) and H_aug.shape == (5, 7)
    x = torch.randn(5, dtype=torch.float64)
    x_prev = torch.randn(5, dtype=torch.float64)
    X = ar2_state(x, x_prev, modes)
    X_next = A_aug @ X
    torch.testing.assert_close(X_next[1], 1.9 * x[1] - 0.95 * x_prev[1])
    torch.testing.assert_close(X_next[3], 1.8 * x[3] - 0.9 * x_prev[3])
    torch.testing.assert_close(X_next[0], 0.5 * x[0] + 0.1 * x[3])
    torch.testing.assert_close(X_next[5:], x[[1, 3]])          # shift register
    torch.testing.assert_close(H_aug @ X, x)


def test_estimate_Q_ar2_recovers_process_noise(generator):
    a1, a2 = resonance([30.0], 1000.0, 0.99)
    q_std, r_std = 0.1, 0.05
    vib = simulate_ar2(a1, a2, q_std, 100000, generator)
    turb = simulate_ar1(torch.tensor([0.95], dtype=torch.float64), q_std, 100000, generator)
    x = torch.cat([vib, turb], dim=1)
    y = x + r_std * torch.randn(x.shape, generator=generator, dtype=torch.float64)
    A_aug, _ = augment_ar2(torch.diag(torch.tensor([0.0, 0.95], dtype=torch.float64)),
                           torch.eye(2, dtype=torch.float64), [0], a1, a2)
    R = r_std**2 * torch.eye(2, dtype=torch.float64)

    Q_aug = estimate_Q_ar2(y, A_aug, [0], R)

    assert Q_aug.shape == (3, 3)
    assert torch.all(Q_aug[2] == 0) and torch.all(Q_aug[:, 2] == 0)
    torch.testing.assert_close(torch.diagonal(Q_aug)[:2],
                               torch.full((2,), q_std**2, dtype=torch.float64),
                               atol=1e-3, rtol=0)


def test_ar2_kalman_predicts_vibration_better_than_ar1(generator):
    # modes 0, 1: vibrations (tip/tilt-like); modes 2, 3: AR(1) turbulence
    fs, delay = 1000.0, 2
    a1, a2 = resonance([17.0, 43.0], fs, 0.995)
    vib = simulate_ar2(a1, a2, 0.05, 30000, generator)
    turb = simulate_ar1(torch.tensor([0.99, 0.95], dtype=torch.float64), 0.1, 30000, generator)
    x = torch.cat([vib, turb], dim=1)
    y = x + 0.05 * torch.randn(x.shape, generator=generator, dtype=torch.float64)
    C = lagged_covariances(y[:20000], max_lag=2)
    gamma = autocovariances(y[:20000], max_lag=20)

    errors = {}
    for name, ar2_modes in [("ar1", None), ("ar2", [0, 1])]:
        model = identify_model(C, A_method="instrumental_variable", ar2_modes=ar2_modes,
                               autocov=gamma)
        dare = steady_state_gain(model.A, model.H, model.Q, model.R,
                                 compute_dtype=torch.float64)
        assert dare.converged
        assert validate_model(model.A, model.Q_unclipped, model.R, dare).ok
        M = build_realtime_matrix(model.A, model.H, dare.K, delay)
        X = torch.zeros(model.A.shape[0], dtype=torch.float64)
        u = []
        for k in range(20000, 30000 - delay):
            X, uk = realtime_step(M, X, y[k])
            u.append(uk)
        u = torch.stack(u)
        # u_k predicts x_{k+delay}; skip the filter transient
        err = x[20000 + delay:][500:] - u[500:]
        errors[name] = err.pow(2).mean(dim=0)

    # vibration modes: much better prediction; turbulence modes: unchanged
    assert torch.all(errors["ar2"][:2] < 0.5 * errors["ar1"][:2])
    torch.testing.assert_close(errors["ar2"][2:], errors["ar1"][2:], rtol=0.05, atol=0)


def test_recursive_covariance_autocovariances(generator):
    x = torch.randn(300, 3, generator=generator, dtype=torch.float64)
    rc = RecursiveCovariance(3, 0.99, max_lag=1, n_autocov_lags=6, dtype=torch.float64)
    rc.update(x[:100])
    rc.update(x[100:])
    g = rc.autocovariances()
    C = rc.covariances()
    assert g.shape == (7, 3)
    torch.testing.assert_close(g[0], torch.diagonal(C[0]))
    torch.testing.assert_close(g[1], torch.diagonal(C[1]))
    lam, N = 0.99, 300
    w = (1 - lam) * lam ** torch.arange(N - 1, -1, -1, dtype=torch.float64)
    torch.testing.assert_close(g[6], (x[6:] * x[:-6] * w[6:, None]).sum(0) / w.sum())


def test_autocovariances_are_lagged_covariance_diagonals(generator):
    x = torch.randn(100, 3, generator=generator, dtype=torch.float64)
    g = autocovariances(x, 3)
    for l, Cl in enumerate(lagged_covariances(x, 3)):
        torch.testing.assert_close(g[l], torch.diagonal(Cl))


def test_estimate_R_residual_form_robust_to_errors_in_A(generator):
    a = torch.tensor([0.995, 0.99, 0.98, 0.95], dtype=torch.float64)
    r_std = 0.02
    x = simulate_ar1(a, 0.1, 100000, generator)
    x = x + r_std * torch.randn(x.shape, generator=generator, dtype=torch.float64)
    A_off = torch.diag(a - 0.005)      # slightly wrong A (e.g. short window, or LS)
    R_true = torch.diag(torch.full((N_MODES,), r_std**2, dtype=torch.float64))

    R_res = estimate_R_from_telemetry(x, A_off, structure="diagonal", form="residual")
    R_lag1 = estimate_R_from_telemetry(x, A_off, structure="diagonal", form="lag1")

    err_res = (torch.diagonal(R_res) - r_std**2).abs()
    err_lag1 = (torch.diagonal(R_lag1) - r_std**2).abs()
    # residual-form bias ~ Sigma * dA * (1 - a^2); lag-1 form bias ~ Sigma * dA
    assert torch.all(err_res < 0.35 * r_std**2)
    assert torch.all(err_lag1 > 3 * err_res)
    # with the exact IV estimate, both forms coincide
    C0, C1, C2 = lagged_covariances(x)
    A_iv = estimate_A_from_covariances(C0, C1, C2, method="instrumental_variable")
    torch.testing.assert_close(
        estimate_R_from_covariances(C0, C1, A_iv, C2=C2, min_fraction=0),
        estimate_R_from_covariances(C0, C1, A_iv, min_fraction=0),
    )


def test_apply_structure(generator):
    B = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    M = B @ B.T
    torch.testing.assert_close(apply_structure(M, "full"), M)
    torch.testing.assert_close(apply_structure(M, "diagonal"), torch.diag(torch.diagonal(M)))
    Mb = apply_structure(M, "block", torch.tensor([0, 0, 1, 1]))
    assert torch.all(Mb[:2, 2:] == 0) and torch.all(Mb[2:, :2] == 0)
    torch.testing.assert_close(Mb[:2, :2], M[:2, :2])
    assert torch.linalg.eigvalsh(Mb).min() >= 0
    with pytest.raises(ValueError):
        apply_structure(M, "block")


def test_identify_model_structure_matches_A(generator):
    x = simulate_linear(coupled_A(), 0.1 * torch.eye(N_MODES, dtype=torch.float64),
                        5000, generator)
    C = lagged_covariances(x)
    m = identify_model(C, A_structure="diagonal")
    for M in (m.A, m.Q, m.R):
        torch.testing.assert_close(M, torch.diag(torch.diagonal(M)))
    m_full = identify_model(C, A_structure="diagonal", Q_structure="full", R_structure="full")
    assert torch.count_nonzero(m_full.Q - torch.diag(torch.diagonal(m_full.Q))) > 0


def test_limit_spectral_radius():
    A = torch.diag(torch.tensor([1.01, 0.5, -1.2, 0.9], dtype=torch.float64))
    Ad = limit_spectral_radius(A, 0.999, "diagonal")
    torch.testing.assert_close(torch.diagonal(Ad),
                               torch.tensor([0.999, 0.5, -0.999, 0.9], dtype=torch.float64))

    Af = coupled_A() * 1.1
    rho = torch.linalg.eigvals(Af).abs().max()
    As = limit_spectral_radius(Af, 0.99, "full")
    torch.testing.assert_close(torch.linalg.eigvals(As).abs().max().to(torch.float64),
                               torch.tensor(0.99, dtype=torch.float64))
    torch.testing.assert_close(As, Af * 0.99 / rho.real.to(torch.float64))

    # block: only the unstable block is scaled
    Ab = torch.zeros(4, 4, dtype=torch.float64)
    Ab[:2, :2] = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float64)
    Ab[2:, 2:] = torch.tensor([[0.5, 0.1], [0.0, 0.4]], dtype=torch.float64)
    Abs = limit_spectral_radius(Ab, 0.99, "block")
    torch.testing.assert_close(Abs[2:, 2:], Ab[2:, 2:])
    torch.testing.assert_close(Abs[:2, :2], 0.99 * Ab[:2, :2])

    stable = 0.5 * torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(limit_spectral_radius(stable, 0.99), stable)
