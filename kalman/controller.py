"""
Adaptive modal Kalman controller for real-time use.

Real-time path (one call to `step` per frame, no host synchronization):

    y_k = z_k / g + u_{k - pol_lag}                 pseudo-open-loop measurement
                                                    (g: WFS gain, per mode)
    [x_{k|k}; u_k^KF; e_k] = M @ [x_{k-1|k-1}; y_k] one GEMV (`build_realtime_matrix`)
    u_k^I = leak * u_{k-1} + gain * z_k             integrator (always computed)
    u_k = alpha * u_k^KF + (1 - alpha) * u_k^I      alpha in {0, 1}
    NIS accumulator += ||e_k||^2                    e_k: whitened innovation

Slow path (between frames, every `update_period` frames): re-identify the
model from the telemetry ring buffer (or from covariances with a forgetting
factor), solve the DARE, validate, and swap the real-time matrix.

Double buffer: M lives in two preallocated buffers. A new model is written
into the inactive one and the active index is flipped between frames, so a
frame never sees a half-written matrix. With CUDA graphs, one graph is
captured per buffer, so the swap is also just the index flip.
"""
from dataclasses import dataclass
from typing import Optional, Union

import torch

from kalman.Kalman_tools import (
    RecursiveCovariance,
    autocovariances,
    build_realtime_matrix,
    identify_ar_model,
    identify_model,
    innovation_whitener,
    lagged_covariances,
    steady_state_gain,
    validate_model,
)


@dataclass
class AdaptiveKalmanConfig:
    """
    Configuration of `AdaptiveKalmanController`.

    n_modes : int
        Number of controlled modes (measurement and command size).
    wfs_gain : float or torch.Tensor
        Gain of the WFS measurement with respect to the command, scalar or
        per mode [n_modes] (e.g. measured with a dither). The filter uses
        z / wfs_gain, so that the pseudo-open-loop measurement is in command
        units; the integrator uses z as given. Overestimating the gain is
        the safe side: if the true gain is rho times wfs_gain, the pseudo-
        open loop rings at ~fs/4 for rho -> 2 and diverges above. A pyramid's
        optical gain rises when the Kalman reduces the residual, so a gain
        measured under the integrator should get a margin (~2x).
    delay : int
        Frames between the command computation and the measurement it
        affects: the physical pseudo-open-loop pairing is
        y_k = z_k + u_{k-delay}. Used for identification.
    pol_lag : int or None
        Pairing fed to the filter, y_k = z_k + u_{k-pol_lag}. Default: delay.
        pol_lag = 1 reproduces SimpleKalman.ipynb (z + z_estimated_kalman).
    command_lead : int or None
        d in u_k = C A^d x_{k|k}. Default: delay.
    window : int
        Telemetry ring-buffer length (frames used per identification).
    update_period : int
        Frames between re-identifications (T_update).
    max_updates : int or None
        Stop re-identifying after this many accepted models (1 = fixed model
        identified once). None: never stop.
    identification : str
        "covariance"
            AR(1) (+ optional AR(2) modes) from lag covariances, with
            Q = Q_raw - R - A R A^T (`identify_model`). Supports
            `forgetting`, A structures and `ar2_modes`, but Q collapses to
            0 (K_ii = 0) in drifting modes of real telemetry.
        "prediction_error"
            Per-mode AR(`ar_order`) models by least squares on the window,
            Q = variance of the one-step prediction errors (positive by
            construction), R = R_fraction * Q (`identify_ar_model`).
            Recommended on the bench. Needs `forgetting=None` and no
            `ar2_modes`; the A_* options other than A_max_radius are unused.
    ar_order : int
        AR order p for "prediction_error" (state size n_modes * p).
    R_fraction : float
        R_ii / Q_ii for "prediction_error" when R_fixed is None. 0.1 suits a
        low-noise WFS (the RAMA bench); when WFS noise dominates the
        one-step prediction error, set R_fixed or a larger fraction.
    forgetting : float or None
        If set, identify from `RecursiveCovariance` with this forgetting
        factor instead of the last `window` frames.
    A_method, A_structure, groups, ridge
        Passed to `estimate_A_from_covariances`.
    A_max_radius : float or None
        Shrink A to this spectral radius (`limit_spectral_radius`) instead
        of rejecting models whose short-window estimate overshoots 1. With
        "prediction_error", the roots of each mode's AR polynomial
        (`limit_ar_radius`).
    R_fixed : torch.Tensor or None
        Known WFS noise covariance [n_modes, n_modes]; estimated from
        telemetry if None.
    R_structure, Q_structure : str
        "full", "diagonal", "block" or "match" (= A_structure), see
        `identify_model`.
    R_min_fraction : float
        Noise floor of the telemetry estimate of R.
    ar2_modes : sequence of int or None
        Modes with an AR(2) model (e.g. (0, 1) for tip and tilt).
    ar2_method : str
        "least_squares" or "instrumental_variable".
    ar2_max_lag : int
        Largest autocovariance lag for the AR(2) instrumental variable.
    max_spectral_radius, max_negative_Q_fraction
        Passed to `validate_model`.
    min_kalman_gain : float or None
        Reject models with K_ii below this in any mode (`validate_model`,
        min_gain): such a mode is never corrected. None: no check.
    dare_tol, dare_max_iter, dare_dtype
        Passed to `steady_state_gain`.
    ident_dtype : torch.dtype
        dtype of the identification (covariances, A, Q, R).
    allow_integrator_fallback : bool
        If True, switch to the integrator when the NIS monitor fires, and
        back when it recovers or a new model is accepted. If False, the
        monitor only logs. (Before the first model, the integrator always
        runs, to collect telemetry.)
    integrator_gain, integrator_leak : float
        u_k = leak * u_{k-1} + gain * z_k (z as given, not divided by
        wfs_gain). integrator_gain may also be a tensor [n_modes].
    nis_window : int
        Frames averaged by the NIS monitor.
    nis_threshold : float
        Fallback when mean(NIS) / n_modes exceeds this (1 = consistent).
    nis_patience : int
        Number of consecutive NIS windows above nis_threshold before the
        fallback (earlier ones only log "nis_alarm"). 2 or more ignores a
        short disturbance burst that the Kalman handles better than the
        integrator.
    nis_recover : float
        Leave the fallback when mean(NIS) / n_modes drops below this
        (recover_on="nis").
    recover_on : str
        "nis": leave the fallback when the NIS, computed while the
        integrator runs, drops below nis_recover. "new_model": only when a
        new model is accepted. Use "new_model" when the model fits the data
        under the integrator but not under the Kalman (e.g. a pyramid whose
        optical gain rises as the residual drops): the NIS is then always
        low under the integrator, and "nis" cycles between both modes.
    use_cuda_graph : bool
        Capture the real-time step as a CUDA graph (CUDA only).
    """
    n_modes: int
    wfs_gain: Union[float, torch.Tensor] = 1.0
    delay: int = 2
    pol_lag: Optional[int] = None
    command_lead: Optional[int] = None
    window: int = 1000
    update_period: int = 500
    max_updates: Optional[int] = None
    identification: str = "covariance"
    ar_order: int = 1
    R_fraction: float = 0.1
    forgetting: Optional[float] = None
    A_method: str = "instrumental_variable"
    A_structure: str = "full"
    groups: Optional[torch.Tensor] = None
    ridge: float = 0.0
    A_max_radius: Optional[float] = 0.9995
    R_fixed: Optional[torch.Tensor] = None
    R_structure: str = "match"
    Q_structure: str = "match"
    R_min_fraction: float = 1e-4
    ar2_modes: Optional[tuple] = None
    ar2_method: str = "instrumental_variable"
    ar2_max_lag: int = 20
    max_spectral_radius: float = 0.9999
    max_negative_Q_fraction: float = 0.2
    min_kalman_gain: Optional[float] = 1e-3
    dare_tol: float = 1e-5
    dare_max_iter: int = 64
    dare_dtype: torch.dtype = torch.float32
    ident_dtype: torch.dtype = torch.float32
    allow_integrator_fallback: bool = False
    integrator_gain: Union[float, torch.Tensor] = 0.3
    integrator_leak: float = 0.995
    nis_window: int = 200
    nis_threshold: float = 2.0
    nis_patience: int = 1
    nis_recover: float = 1.3
    recover_on: str = "nis"
    use_cuda_graph: bool = False


class AdaptiveKalmanController:
    """
    Steady-state Kalman controller with periodic re-identification, model
    validation, double-buffered real-time matrix and an innovation (NIS)
    monitor with optional integrator fallback.

    Parameters
    ----------
    config : AdaptiveKalmanConfig
    device : torch.device or str
    dtype : torch.dtype
        dtype of the real-time path (float32 on GPU).

    Attributes
    ----------
    history : list of dict
        Events: {"frame", "event", ...} with event in "accepted", "rejected",
        "fallback", "recovered", "nis_alarm".
    nis_log : list of (frame, mean NIS / n_modes)
    model, dare, check
        Last accepted IdentifiedModel, DAREResult and ModelCheck.
    mode : str
        "bootstrap" (no model yet), "kalman" or "integrator".
    """

    def __init__(self, config, device, dtype=torch.float32):
        c = config
        self.config = c
        self.device = torch.device(device)
        self.dtype = dtype
        self.pol_lag = c.delay if c.pol_lag is None else c.pol_lag
        self.command_lead = c.delay if c.command_lead is None else c.command_lead
        if c.window < c.update_period:
            raise ValueError("window must be >= update_period")

        n = c.n_modes
        self.ar2 = None
        if c.ar2_modes is not None and len(c.ar2_modes) > 0:
            self.ar2 = torch.as_tensor(c.ar2_modes, dtype=torch.long, device=self.device)
        if c.identification == "prediction_error":
            if c.forgetting is not None or self.ar2 is not None:
                raise ValueError('identification="prediction_error" needs forgetting=None '
                                 'and no ar2_modes (use ar_order)')
            if c.ar_order < 1:
                raise ValueError("ar_order must be >= 1")
            self.ns = n * c.ar_order
        elif c.identification == "covariance":
            self.ns = n + (0 if self.ar2 is None else self.ar2.numel())
        else:
            raise ValueError(f"Unknown identification {c.identification!r}; "
                             "use 'covariance' or 'prediction_error'")
        self.n = n
        if c.recover_on not in ("nis", "new_model"):
            raise ValueError(f"Unknown recover_on {c.recover_on!r}; use 'nis' or 'new_model'")

        kw = dict(device=self.device, dtype=dtype)
        gain = torch.as_tensor(c.wfs_gain, **kw)
        if bool((gain <= 0).any()):
            raise ValueError("wfs_gain must be > 0")
        self.inv_wfs_gain = (1 / gain).expand(n).clone()    # [n]
        self.int_gain = torch.as_tensor(c.integrator_gain, **kw)
        rows, cols = self.ns + 2 * n, self.ns + n
        self._M = [torch.zeros(rows, cols, **kw), torch.zeros(rows, cols, **kw)]
        self._active = 0

        # real-time state (all preallocated, updated in place)
        self.z = torch.zeros(n, **kw)
        self.v = torch.zeros(cols, **kw)                     # [x_{k-1|k-1}; y_k]
        self.out = torch.zeros(rows, **kw)                   # [x_{k|k}; u_k; e_k]
        self.U = torch.zeros(max(c.delay, self.pol_lag), n, **kw)   # U[j] = u_{k-1-j}
        self.u_out = torch.zeros(n, **kw)
        self.T = torch.zeros(c.window, n, **kw)              # physical POL telemetry
        self.T_idx = torch.zeros(1, device=self.device, dtype=torch.long)
        self.alpha = torch.zeros(1, **kw)                    # 1: Kalman, 0: integrator
        self.nis_sum = torch.zeros(1, **kw)

        self.frame = 0
        self.mode = "bootstrap"
        self.n_accepted = 0
        self.model = self.dare = self.check = None
        self.history = []
        self.nis_log = []
        self._nis_frames = 0
        self._nis_high = 0                                   # consecutive windows above threshold

        self.rcov = None
        if c.forgetting is not None:
            self.rcov = RecursiveCovariance(
                n, c.forgetting, max_lag=2,
                n_autocov_lags=c.ar2_max_lag if self.ar2 is not None else 0,
                device=self.device, dtype=c.ident_dtype,
            )
            self._fed = 0

        self._graphs = None
        if c.use_cuda_graph:
            if self.device.type != "cuda":
                raise ValueError("use_cuda_graph needs a CUDA device")
            self._capture_graphs()

    # ---------------------------------------------------------
    # Real-time path
    # ---------------------------------------------------------

    def _rt_body(self, M):
        c, n, ns = self.config, self.n, self.ns
        U = self.U
        z_cmd = self.z * self.inv_wfs_gain                  # residual in command units
        y_filter = z_cmd + U[self.pol_lag - 1]
        y_ident = z_cmd + U[c.delay - 1]
        self.v[ns:].copy_(y_filter)
        torch.mv(M, self.v, out=self.out)
        self.v[:ns].copy_(self.out[:ns])
        u_int = c.integrator_leak * U[0] + self.int_gain * self.z
        u = torch.lerp(u_int, self.out[ns:ns + n], self.alpha)
        for j in range(U.shape[0] - 1, 0, -1):
            U[j].copy_(U[j - 1])
        U[0].copy_(u)
        self.u_out.copy_(u)
        self.T.index_copy_(0, self.T_idx, y_ident.unsqueeze(0))
        self.T_idx.add_(1).remainder_(c.window)
        e = self.out[ns + n:]
        self.nis_sum.add_((e * e).sum())

    def _state_tensors(self):
        return [self.z, self.v, self.out, self.U, self.u_out, self.T,
                self.T_idx, self.nis_sum]

    def _capture_graphs(self):
        saved = [t.clone() for t in self._state_tensors()]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for M in self._M:
                for _ in range(3):
                    self._rt_body(M)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._graphs = []
        for M in self._M:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._rt_body(M)
            self._graphs.append(g)
        for t, t0 in zip(self._state_tensors(), saved):
            t.copy_(t0)

    def rt_step(self, z):
        """
        Real-time part of a frame only (no slow path): the command for z.

        Parameters
        ----------
        z : torch.Tensor
            WFS modal measurement of the residual, shape [n_modes].

        Returns
        -------
        u : torch.Tensor
            Command, shape [n_modes]. A view on an internal buffer that the
            next call overwrites: copy it if you keep it.
        """
        self.z.copy_(z.reshape(-1))
        if self._graphs is not None:
            self._graphs[self._active].replay()
        else:
            self._rt_body(self._M[self._active])
        self.frame += 1
        self._nis_frames += 1
        return self.u_out

    def step(self, z):
        """
        One frame: real-time step, then (between frames) the NIS monitor and
        re-identification when they are due.

        Parameters
        ----------
        z : torch.Tensor
            WFS modal measurement of the residual, shape [n_modes].

        Returns
        -------
        u : torch.Tensor
            Command, shape [n_modes] (internal buffer, see `rt_step`).
        """
        u = self.rt_step(z)
        self.slow_path()
        return u

    # ---------------------------------------------------------
    # Slow path
    # ---------------------------------------------------------

    def slow_path(self):
        """Run the NIS monitor and re-identification if due at this frame."""
        c = self.config
        if self.mode != "bootstrap" and self._nis_frames >= c.nis_window:
            self._check_nis()
        done = c.max_updates is not None and self.n_accepted >= c.max_updates
        if (not done and self.frame >= c.window
                and (self.frame - c.window) % c.update_period == 0):
            self.reidentify()

    def telemetry(self):
        """
        Pseudo-open-loop telemetry y_k = z_k + u_{k-delay}, oldest first.

        Returns
        -------
        y : torch.Tensor
            Shape [min(frame, window), n_modes].
        """
        W = self.config.window
        i = int(self.T_idx)
        if self.frame < W:
            return self.T[:self.frame].clone()
        return torch.cat([self.T[i:], self.T[:i]])

    def _covariances(self):
        c = self.config
        y = self.telemetry().to(c.ident_dtype)
        if self.rcov is None:
            C = lagged_covariances(y, max_lag=2)
            g = autocovariances(y, c.ar2_max_lag) if self.ar2 is not None else None
            return C, g
        new = min(self.frame - self._fed, y.shape[0])
        self.rcov.update(y[y.shape[0] - new:])
        self._fed = self.frame
        g = self.rcov.autocovariances() if self.ar2 is not None else None
        return self.rcov.covariances(), g

    def reidentify(self):
        """
        Identify a model from the current telemetry, solve the DARE,
        validate, and swap the real-time matrix if the model is valid.

        Returns
        -------
        accepted : bool
        """
        c = self.config
        try:
            if c.identification == "prediction_error":
                y = self.telemetry().to(c.ident_dtype)
                model = identify_ar_model(
                    y, c.ar_order, max_radius=c.A_max_radius,
                    R=None if c.R_fixed is None else c.R_fixed.to(y),
                    R_fraction=c.R_fraction,
                )
            else:
                C, g = self._covariances()
                model = identify_model(
                    C, A_method=c.A_method, A_structure=c.A_structure,
                    groups=c.groups, ridge=c.ridge, A_max_radius=c.A_max_radius,
                    R=None if c.R_fixed is None else c.R_fixed.to(C[0]),
                    R_structure=c.R_structure, Q_structure=c.Q_structure,
                    R_min_fraction=c.R_min_fraction,
                    ar2_modes=self.ar2, ar2_method=c.ar2_method, autocov=g,
                )
            dare = steady_state_gain(model.A, model.H, model.Q, model.R,
                                     tol=c.dare_tol, max_iter=c.dare_max_iter,
                                     compute_dtype=c.dare_dtype, on_fail="ignore")
            check = validate_model(model.A, model.Q_unclipped, model.R, dare,
                                   max_spectral_radius=c.max_spectral_radius,
                                   max_negative_Q_fraction=c.max_negative_Q_fraction,
                                   min_gain=c.min_kalman_gain)
        except (RuntimeError, torch.linalg.LinAlgError) as err:
            self.history.append({"frame": self.frame, "event": "rejected",
                                 "reasons": (f"identification failed: {err}",)})
            return False

        if not check.ok:
            self.history.append({"frame": self.frame, "event": "rejected",
                                 "reasons": check.reasons,
                                 "spectral_radius": check.spectral_radius})
            return False

        W = innovation_whitener(model.H, dare.P, model.R)
        M = build_realtime_matrix(model.A, model.H, dare.K, self.command_lead, whiten=W)
        self.load_matrix(M)
        self.model, self.dare, self.check = model, dare, check
        self.n_accepted += 1
        self.history.append({"frame": self.frame, "event": "accepted",
                             "spectral_radius": check.spectral_radius,
                             "negative_Q_fraction": check.negative_Q_fraction,
                             "dare_iterations": dare.iterations})
        return True

    def load_matrix(self, M):
        """
        Double-buffered swap: write M into the inactive buffer, then make it
        active. Called between frames, so every frame uses one whole matrix.
        On the first model, the state is initialized from the telemetry and
        the Kalman command is switched on.

        Parameters
        ----------
        M : torch.Tensor
            From `build_realtime_matrix` with whitening rows,
            shape [N_state + 2 n_modes, N_state + n_modes].
        """
        inactive = 1 - self._active
        self._M[inactive].copy_(M)
        self._active = inactive
        self.nis_sum.zero_()
        self._nis_frames = 0
        self._nis_high = 0
        if self.mode != "kalman":
            if self.mode == "bootstrap":
                self._init_state()
            else:
                self.history.append({"frame": self.frame, "event": "recovered"})
            self.alpha.fill_(1.0)
            self.mode = "kalman"

    def _init_state(self):
        y = self.telemetry()
        if y.shape[0] == 0:
            return
        n = self.n
        self.v[:n].copy_(y[-1])
        if self.ar2 is not None and y.shape[0] > 1:
            self.v[n:self.ns].copy_(y[-2][self.ar2])
        if self.config.identification == "prediction_error":
            # stacked state [x_k; x_{k-1}; ...; x_{k-p+1}]
            for j in range(1, min(self.config.ar_order, y.shape[0])):
                self.v[j * n:(j + 1) * n].copy_(y[-1 - j])

    def _check_nis(self):
        """
        End of a monitor window (`nis_window` frames): log the mean NIS.
        Fall back to the integrator after `nis_patience` windows in a row
        with NIS / n_modes > nis_threshold. With recover_on="nis", go back
        to the Kalman when the NIS (computed while the integrator runs)
        drops below nis_recover; with "new_model", only when a new model is
        accepted (`load_matrix`).
        """
        c = self.config
        nis = float(self.nis_sum) / (self._nis_frames * self.n)
        self.nis_sum.zero_()
        self._nis_frames = 0
        self.nis_log.append((self.frame, nis))
        if self.mode == "kalman" and nis > c.nis_threshold:
            self._nis_high += 1
            if c.allow_integrator_fallback and self._nis_high >= c.nis_patience:
                self.alpha.fill_(0.0)
                self.mode = "integrator"
                self._nis_high = 0
                self.history.append({"frame": self.frame, "event": "fallback", "nis": nis})
            else:
                self.history.append({"frame": self.frame, "event": "nis_alarm", "nis": nis})
        elif self.mode == "kalman":
            self._nis_high = 0
        elif self.mode == "integrator" and c.recover_on == "nis" and nis < c.nis_recover:
            self.alpha.fill_(1.0)
            self.mode = "kalman"
            self._nis_high = 0
            self.history.append({"frame": self.frame, "event": "recovered", "nis": nis})


class IntegratorController:
    """
    Leaky integrator with the same `step` interface:

        u_k = leak * u_{k-1} + gain * z_k

    Parameters
    ----------
    n_modes : int
    gain, leak : float
    device, dtype
    """

    def __init__(self, n_modes, gain=0.3, leak=0.995, device="cpu", dtype=torch.float32):
        self.gain, self.leak = gain, leak
        self.u = torch.zeros(n_modes, device=device, dtype=dtype)

    def step(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor
            Residual measurement, shape [n_modes].

        Returns
        -------
        u : torch.Tensor
            Command, shape [n_modes].
        """
        self.u = self.leak * self.u + self.gain * z.reshape(-1)
        return self.u
