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


def true_model_controller(a, q_std, r_std, **kw):
    """Controller running the true steady-state model from frame 0 (kw: config options)."""
    A = torch.diag(torch.as_tensor(a, dtype=DTYPE))
    I = torch.eye(A.shape[0], dtype=DTYPE)
    dare = steady_state_gain(A, I, q_std**2 * I, r_std**2 * I, compute_dtype=DTYPE)
    M = build_realtime_matrix(A, I, dare.K, 2, whiten=innovation_whitener(I, dare.P, r_std**2 * I))
    ctrl = AdaptiveKalmanController(config(max_updates=0, **kw), "cpu", DTYPE)
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
    # mechanics only: the covariance model of this window has K_ii = 0 in
    # one mode (rejected by default), but any accepted matrix will do here
    ctrl = AdaptiveKalmanController(config(max_updates=1, min_kalman_gain=None), "cpu", DTYPE)
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


@pytest.mark.parametrize("patience", [1, 2])
def test_nis_patience_ignores_a_short_burst(patience):
    phi, noise = make_atmosphere(4000, A_SLOW, 0.1, 0.3)
    noise[3000:3200] *= 5        # one NIS window (frames 3000-3200) of bad data
    ctrl = AdaptiveKalmanController(
        config(max_updates=1, min_kalman_gain=None, allow_integrator_fallback=True,
               nis_patience=patience), "cpu", DTYPE)
    run_loop(ctrl, phi, noise)
    events = [h["event"] for h in ctrl.history]
    assert "nis_alarm" in events or "fallback" in events
    if patience == 1:
        assert "fallback" in events
    else:
        assert "fallback" not in events and ctrl.mode == "kalman"


def end_window(ctrl, nis):
    """Close a monitor window whose mean NIS / n_modes is `nis`."""
    ctrl.nis_sum.fill_(nis * ctrl.config.nis_window * ctrl.n)
    ctrl._nis_frames = ctrl.config.nis_window
    ctrl._check_nis()
    return ctrl.history[-1]["event"] if ctrl.history else None


def test_nis_patience_counts_again_after_recovery():
    ctrl = true_model_controller(A_SLOW, 0.1, 0.3, allow_integrator_fallback=True, nis_patience=2)
    events = []
    for nis in (5.0, 5.0, 0.5, 5.0, 5.0):
        n_before = len(ctrl.history)
        end_window(ctrl, nis)
        events.append(ctrl.history[-1]["event"] if len(ctrl.history) > n_before else None)
    # high, high -> fallback; low under the integrator -> recovered;
    # then patience again: one alarm before the next fallback
    assert events == ["nis_alarm", "fallback", "recovered", "nis_alarm", "fallback"]


def test_recover_on_new_model_only():
    ctrl = true_model_controller(A_SLOW, 0.1, 0.3, allow_integrator_fallback=True,
                                 recover_on="new_model")
    M = ctrl._M[ctrl._active].clone()
    end_window(ctrl, 5.0)
    assert ctrl.mode == "integrator"
    end_window(ctrl, 0.1)                 # low NIS under the integrator: no recovery
    assert ctrl.mode == "integrator" and float(ctrl.alpha) == 0.0
    ctrl.load_matrix(M)                   # a new model
    assert ctrl.mode == "kalman" and ctrl.history[-1]["event"] == "recovered"
    with pytest.raises(ValueError):
        AdaptiveKalmanController(config(recover_on="other"), "cpu", DTYPE)


def run_loop_kalman_gain(ctrl, phi, noise, rho):
    """run_loop where the WFS gain is rho while the Kalman controls (a pyramid
    whose optical gain rises as the residual drops) and 1 otherwise."""
    u1 = torch.zeros(phi.shape[1], dtype=DTYPE)
    u2 = torch.zeros(phi.shape[1], dtype=DTYPE)
    res = []
    for k in range(2, phi.shape[0]):
        res.append(phi[k] - u1)
        g = rho if ctrl.mode == "kalman" else 1.0
        z = g * (phi[k - 1] - u2 + noise[k])
        u = ctrl.step(z).clone()
        u2, u1 = u1, u
    return torch.stack(res)


def test_gain_margin_keeps_kalman_stable_when_wfs_gain_rises():
    # the WFS gain doubles once the Kalman controls: with wfs_gain at the
    # gain measured under the integrator the pseudo-open loop rings (high
    # NIS, worse than the integrator); with a 2x margin it stays far better
    phi, noise = make_atmosphere(8000, [0.999, 0.998, 0.995, 0.99], 0.05, 0.01)
    res_int = run_loop(IntegratorController(N_MODES, 0.2, 0.99, dtype=DTYPE), phi, noise)
    out = {}
    for margin in (1.0, 2.0):
        ctrl = AdaptiveKalmanController(
            config(identification="prediction_error", max_updates=1, integrator_gain=0.2,
                   wfs_gain=margin), "cpu", DTYPE)
        res = run_loop_kalman_gain(ctrl, phi, noise, 2.0)
        out[margin] = (res[3000:].pow(2).mean() / res_int[3000:].pow(2).mean(), ctrl)
    assert out[1.0][0] > 1.5
    assert out[2.0][0] < 0.7 and out[2.0][1].mode == "kalman"


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


@pytest.mark.parametrize("order", [1, 3])
def test_prediction_error_identification_close_to_true_model(order):
    # low WFS noise, as on the bench: R = R_fraction * Q is then a good model
    phi, noise = make_atmosphere(6000, A_SLOW, 0.1, 0.05)
    ctrl = AdaptiveKalmanController(
        config(identification="prediction_error", ar_order=order), "cpu", DTYPE)
    res_kf = run_loop(ctrl, phi, noise)
    res_true = run_loop(true_model_controller(A_SLOW, 0.1, 0.05), phi, noise)
    assert ctrl.history[0] == {**ctrl.history[0], "frame": 2000, "event": "accepted"}
    assert ctrl.mode == "kalman" and ctrl.model.A.shape == (N_MODES * order, N_MODES * order)
    assert res_kf[3000:].pow(2).mean() < 1.05 * res_true[3000:].pow(2).mean()
    assert all(0.7 < nis < 1.3 for _, nis in ctrl.nis_log[-5:])


def run_loop_gain(ctrl, phi, noise, gain):
    """run_loop with a WFS of gain `gain` (per mode): z = gain * (phi_{k-1} - u_{k-2} + n_k)."""
    u1 = torch.zeros(phi.shape[1], dtype=DTYPE)
    u2 = torch.zeros(phi.shape[1], dtype=DTYPE)
    res = []
    for k in range(2, phi.shape[0]):
        res.append(phi[k] - u1)
        z = gain * (phi[k - 1] - u2 + noise[k])
        u = ctrl.step(z).clone()
        u2, u1 = u1, u
    return torch.stack(res)


@pytest.mark.parametrize("identification", ["covariance", "prediction_error"])
def test_wfs_gain_normalization(identification):
    # a WFS of gain g with wfs_gain=g (and the integrator gain divided by g)
    # is the same loop as a WFS of gain 1
    phi, noise = make_atmosphere(4000, A_SLOW, 0.1, 0.05)
    g = torch.tensor([0.2, 0.25, 0.3, 0.45], dtype=DTYPE)
    kw = dict(identification=identification, ar_order=2 if identification == "prediction_error" else 1)
    ctrl_g = AdaptiveKalmanController(
        config(wfs_gain=g, integrator_gain=0.4 / g, **kw), "cpu", DTYPE)
    ctrl_1 = AdaptiveKalmanController(config(**kw), "cpu", DTYPE)
    res_g = run_loop_gain(ctrl_g, phi, noise, g)
    res_1 = run_loop(ctrl_1, phi, noise)
    assert ctrl_g.mode == "kalman"
    torch.testing.assert_close(res_g, res_1, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(ctrl_g.telemetry(), ctrl_1.telemetry(), atol=1e-8, rtol=1e-6)


def test_prediction_error_option_checks():
    with pytest.raises(ValueError):
        AdaptiveKalmanController(
            config(identification="prediction_error", forgetting=0.999), "cpu", DTYPE)
    with pytest.raises(ValueError):
        AdaptiveKalmanController(
            config(identification="prediction_error", ar2_modes=(0,)), "cpu", DTYPE)
    with pytest.raises(ValueError):
        AdaptiveKalmanController(config(identification="other"), "cpu", DTYPE)
    with pytest.raises(ValueError):
        AdaptiveKalmanController(config(wfs_gain=0.0), "cpu", DTYPE)


def test_zero_gain_mode_rejected_by_default():
    # this window gives the covariance model K_ii = 0 in mode 1 (Q collapse)
    phi, noise = make_atmosphere(3000, A_SLOW, 0.1, 0.3)
    ctrl = AdaptiveKalmanController(config(max_updates=1), "cpu", DTYPE)
    run_loop(ctrl, phi, noise, stop=2500)
    assert ctrl.history[0]["event"] == "rejected"
    assert "K_ii" in ctrl.history[0]["reasons"][-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_matches_eager():
    phi, noise = make_atmosphere(3000, A_SLOW, 0.1, 0.3)
    outs = []
    for graph in (False, True):
        cfg = config(use_cuda_graph=graph, ident_dtype=torch.float32,
                     dare_dtype=torch.float32, min_kalman_gain=None)   # mechanics only
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
