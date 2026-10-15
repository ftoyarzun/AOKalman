import pytest
import torch

from kalman.Kalman_tools import (
    build_realtime_matrix,
    innovation_whitener,
    realtime_step,
    steady_state_gain,
)
from kalman.controller import (
    AdaptiveKalmanConfig,
    AdaptiveKalmanController,
    IntegratorController,
)

N_MODES = 4
DTYPE = torch.float64


def make_atmosphere(n_frames, a, q_std, r_std, seed=0, q_change=None):
    """
    AR(1) 'atmosphere' phi and white WFS noise n, float64 on CPU.
    q_change = (frame, factor) multiplies the process noise from that frame.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.as_tensor(a, dtype=DTYPE)
    phi = torch.zeros(n_frames, a.numel(), dtype=DTYPE)
    w = q_std * torch.randn(n_frames, a.numel(), generator=g, dtype=DTYPE)
    if q_change is not None:
        w[q_change[0]:] *= q_change[1]
    for k in range(n_frames - 1):
        phi[k + 1] = a * phi[k] + w[k]
    noise = r_std * torch.randn(n_frames, a.numel(), generator=g, dtype=DTYPE)
    return phi, noise


def run_loop(ctrl, phi, noise, start=2, stop=None):
    """
    Closed loop with the timing of SimpleKalman.ipynb (2-frame delay):
    z_k = phi_{k-1} - u_{k-2} + n_k, residual_k = phi_k - u_{k-1}.
    Returns the residuals, shape [frames, N_modes].
    """
    stop = phi.shape[0] if stop is None else stop
    u1 = torch.zeros(phi.shape[1], dtype=DTYPE)   # u_{k-1}
    u2 = torch.zeros(phi.shape[1], dtype=DTYPE)   # u_{k-2}
    res = []
    for k in range(start, stop):
        res.append(phi[k] - u1)          # the DM holds u_{k-1} during frame k
        z = phi[k - 1] - u2 + noise[k]
        u = ctrl.step(z).clone()
        u2, u1 = u1, u
    return torch.stack(res)


def true_model_controller(a, q_std, r_std):
    """Controller running the true steady-state model from frame 0."""
    A = torch.diag(torch.as_tensor(a, dtype=DTYPE))
    I = torch.eye(A.shape[0], dtype=DTYPE)
    dare = steady_state_gain(A, I, q_std**2 * I, r_std**2 * I, compute_dtype=DTYPE)
    M = build_realtime_matrix(A, I, dare.K, 2, whiten=innovation_whitener(I, dare.P, r_std**2 * I))
    ctrl = AdaptiveKalmanController(config(max_updates=0), "cpu", DTYPE)
    ctrl.load_matrix(M)
    return ctrl


def config(**kw):
    # diagonal IV A: well conditioned on 2000 frames, so these tests check
    # the controller mechanics rather than the estimator variance
    base = dict(n_modes=N_MODES, delay=2, window=2000, update_period=1000,
                ident_dtype=DTYPE, dare_dtype=DTYPE, integrator_gain=0.4,
                integrator_leak=0.99, nis_window=200, A_structure="diagonal")
    base.update(kw)
    return AdaptiveKalmanConfig(**base)


A_SLOW = [0.995, 0.99, 0.98, 0.95]


def test_bootstraps_then_kalman_beats_integrator():
    phi, noise = make_atmosphere(8000, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(), "cpu", DTYPE)
    res_kf = run_loop(ctrl, phi, noise)
    res_int = run_loop(IntegratorController(N_MODES, 0.4, 0.99, dtype=DTYPE), phi, noise)

    accepted = [h for h in ctrl.history if h["event"] == "accepted"]
    assert accepted[0]["frame"] == 2000 and len(accepted) >= 5
    assert ctrl.mode == "kalman"
    # both identical until the first model
    torch.testing.assert_close(res_kf[:1990], res_int[:1990])
    assert res_kf[3000:].pow(2).mean() < 0.8 * res_int[3000:].pow(2).mean()
    # a consistent model: NIS / n_modes ~ 1
    assert all(0.7 < nis < 1.5 for _, nis in ctrl.nis_log[-5:])


def test_true_model_is_consistent():
    phi, noise = make_atmosphere(4000, A_SLOW, 0.1, 0.3)
    ctrl = true_model_controller(A_SLOW, 0.1, 0.3)
    run_loop(ctrl, phi, noise)
    assert all(0.85 < nis < 1.15 for _, nis in ctrl.nis_log)


@pytest.mark.parametrize("structure", ["diagonal", "full"])
def test_identified_close_to_true_model(structure):
    phi, noise = make_atmosphere(6000, A_SLOW, 0.1, 0.05)
    ctrl = AdaptiveKalmanController(config(A_structure=structure), "cpu", DTYPE)
    res_kf = run_loop(ctrl, phi, noise)
    res_true = run_loop(true_model_controller(A_SLOW, 0.1, 0.05), phi, noise)
    assert ctrl.history[0]["event"] == "accepted" and ctrl.mode == "kalman"
    assert res_kf[3000:].pow(2).mean() < 1.1 * res_true[3000:].pow(2).mean()


def test_telemetry_is_physical_pseudo_open_loop():
    phi, noise = make_atmosphere(600, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(window=500, update_period=500, max_updates=0),
                                    "cpu", DTYPE)
    run_loop(ctrl, phi, noise)
    # frames k = 100 .. 599 went in; y_k = z_k + u_{k-2} = phi_{k-1} + n_k
    torch.testing.assert_close(ctrl.telemetry(), phi[99:599] + noise[100:600])


def test_kalman_path_equals_realtime_step():
    phi, noise = make_atmosphere(3000, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(max_updates=1), "cpu", DTYPE)
    run_loop(ctrl, phi, noise, stop=2500)
    m = ctrl.model
    M = build_realtime_matrix(m.A, m.H, ctrl.dare.K, 2)
    x = ctrl.v[:N_MODES].clone()
    y_prev = ctrl.U.clone()
    z = torch.randn(N_MODES, dtype=DTYPE)
    x_ref, u_ref = realtime_step(M, x, z + y_prev[1])
    u = ctrl.step(z)
    torch.testing.assert_close(u, u_ref)
    torch.testing.assert_close(ctrl.v[:N_MODES], x_ref)


def test_double_buffer_swap():
    phi, noise = make_atmosphere(4500, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(), "cpu", DTYPE)
    run_loop(ctrl, phi, noise, stop=2002)
    first = ctrl._active
    M_first = ctrl._M[first].clone()
    run_loop(ctrl, phi, noise, start=2002, stop=3002)
    assert ctrl._active == 1 - first
    torch.testing.assert_close(ctrl._M[first], M_first)   # old buffer untouched
    assert not torch.equal(ctrl._M[ctrl._active], M_first)


def test_invalid_model_rejected_and_integrator_kept():
    phi, noise = make_atmosphere(3000, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(max_spectral_radius=0.5), "cpu", DTYPE)
    res = run_loop(ctrl, phi, noise)
    res_int = run_loop(IntegratorController(N_MODES, 0.4, 0.99, dtype=DTYPE), phi, noise)
    assert ctrl.mode == "bootstrap" and ctrl.model is None
    assert ctrl.history[0]["event"] == "rejected"
    assert "spectral radius" in ctrl.history[0]["reasons"][0]
    torch.testing.assert_close(res, res_int)


@pytest.mark.parametrize("allow", [True, False])
def test_nis_monitor_fallback_switch(allow):
    phi, noise = make_atmosphere(5000, A_SLOW, 0.1, 0.3)
    noise[3000:] *= 5            # conditions change: noise 25x larger in variance
    ctrl = AdaptiveKalmanController(
        config(max_updates=1, allow_integrator_fallback=allow), "cpu", DTYPE)
    run_loop(ctrl, phi, noise)
    events = [h["event"] for h in ctrl.history]
    if allow:
        assert "fallback" in events and ctrl.mode == "integrator"
        assert float(ctrl.alpha) == 0.0
    else:
        assert "fallback" not in events and "nis_alarm" in events
        assert ctrl.mode == "kalman"


def test_no_model_accepted_when_unidentifiable():
    # SNR r/q = 225 after frame 3000: 2000 frames cannot separate Q from R,
    # so the new models are rejected and the fallback keeps the integrator
    phi, noise = make_atmosphere(9000, A_SLOW, 0.1, 0.3)
    noise[3000:] *= 5
    ctrl = AdaptiveKalmanController(
        config(allow_integrator_fallback=True, nis_threshold=1.5), "cpu", DTYPE)
    run_loop(ctrl, phi, noise)
    events = [h["event"] for h in ctrl.history]
    after = events[events.index("fallback"):]
    assert "accepted" not in after and ctrl.mode == "integrator"


def test_fallback_recovers_on_new_model():
    # seeing degrades (process noise x4) at frame 3000, and stays identifiable
    phi, noise = make_atmosphere(9000, A_SLOW, 0.1, 0.3, q_change=(3000, 4.0))
    ctrl = AdaptiveKalmanController(
        config(allow_integrator_fallback=True, nis_threshold=1.5), "cpu", DTYPE)
    run_loop(ctrl, phi, noise)
    events = [h["event"] for h in ctrl.history]
    assert "fallback" in events
    assert "recovered" in events[events.index("fallback"):]
    assert ctrl.mode == "kalman"


def test_forgetting_factor_mode():
    phi, noise = make_atmosphere(6000, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(forgetting=1 - 1 / 1500), "cpu", DTYPE)
    res = run_loop(ctrl, phi, noise)
    res_int = run_loop(IntegratorController(N_MODES, 0.4, 0.99, dtype=DTYPE), phi, noise)
    assert sum(h["event"] == "accepted" for h in ctrl.history) >= 3
    assert res[3000:].pow(2).mean() < 0.8 * res_int[3000:].pow(2).mean()


def test_ar2_modes_and_pol_lag_options():
    phi, noise = make_atmosphere(4000, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(ar2_modes=(0, 1)), "cpu", DTYPE)
    run_loop(ctrl, phi, noise)
    assert ctrl.model.A.shape == (N_MODES + 2, N_MODES + 2)
    assert ctrl.mode == "kalman"

    ctrl1 = AdaptiveKalmanController(config(pol_lag=1, command_lead=1), "cpu", DTYPE)
    run_loop(ctrl1, phi, noise)
    assert ctrl1.mode == "kalman"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_matches_eager():
    phi, noise = make_atmosphere(3000, A_SLOW, 0.1, 0.3)
    outs = []
    for graph in (False, True):
        cfg = config(use_cuda_graph=graph, ident_dtype=torch.float32,
                     dare_dtype=torch.float32)
        ctrl = AdaptiveKalmanController(cfg, "cuda", torch.float32)
        u1 = torch.zeros(N_MODES, device="cuda")
        u2 = torch.zeros(N_MODES, device="cuda")
        us = []
        for k in range(2, 3000):
            z = (phi[k - 1] + noise[k]).float().cuda() - u2
            u = ctrl.step(z).clone()
            u2, u1 = u1, u
            us.append(u)
        outs.append(torch.stack(us).cpu())
        assert ctrl.mode == "kalman"
    torch.testing.assert_close(outs[0], outs[1], atol=1e-5, rtol=1e-4)
