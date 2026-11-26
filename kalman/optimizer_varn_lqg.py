"""VARN-LQG -- the optimizer: fits a joint VAR(N) model of all the modes and its Kalman predictor.

On the pseudo-open-loop telemetry (the turbulence, `da.telemetry.pol`):
  1. the VAR(N) model pol_t+1 = var @ [pol_t, ..., pol_t-N+1], by least squares; Q, the
     covariance of what it does not predict, from its one-step prediction residuals
  2. R, the WFS's measurement-noise covariance: the noise (photon + read-out, propagated through
     the reconstructor) plus the aliasing, from R_noise.npy and R_aliasing.npy (made offline by
     make_R.py), in the units of the pseudo open loop: / og^2
  3. the steady-state Kalman gain of its state-space form (iterated Riccati equation)
  4. the prediction `delay` frames ahead: the first n_modes rows of A^delay
then sends them to the controller. N is the controller's: the size of its params.

With config "learn" (the default, "all"), from the second fit on, the predictor is then learnt
end to end (learned_predictor.py): its matrices are trained to minimise the residual it would
leave on the telemetry, starting from whichever of the model above and the predictor in place
does better on the held-out frames. It is sent only if it beats the predictor in place there.
Needs torch; without it, the model alone, as with "learn": "off".

    RAMA_CUP_DA=socket:ipc:///tmp/cup-varn_lqg.opt.sock python3 optimizer_varn_lqg.py
"""
import os

# Not real time, and on the same machine as the loop: one thread, low priority, so that the
# fit never takes the CPU from the controller and the server.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
if hasattr(os, "nice"):
    os.nice(10)

import numpy as np  # noqa: E402  (after the thread settings)
import ramacup  # noqa: E402
from varn_model import ahead, companion, fit_var, kalman_gain, split_delay  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))

try:                                    # the learnt predictor needs torch (CPU)
    import learned_predictor  # noqa: E402
    LEARN_ERROR = None
except ImportError as error:            # torch missing: the model-based predictor only
    learned_predictor, LEARN_ERROR = None, error


def load_R(n, config):
    """
    The WFS's measurement-noise covariance, in the units of da.input:

        R = noise_scale * R_noise + aliasing_scale * R_aliasing

    from the files of config (made by make_R.py, in this folder), cut to the first n modes.

    Returns
    -------
    R : ndarray [n, n] or None
        None when a file is missing or too small for n modes.
    why : str
        Where R comes from, or why it is None.
    """
    parts = []
    for file_key, scale_key in (("r_noise_file", "noise_scale"), ("r_aliasing_file", "aliasing_scale")):
        path = os.path.join(HERE, config[file_key])
        if not os.path.exists(path):
            return None, f"{config[file_key]} not found"
        R = np.load(path).astype(float)
        if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] < n:
            return None, f"{config[file_key]}: shape {R.shape}, needs at least ({n}, {n})"
        parts.append(config[scale_key] * R[:n, :n])
    R = parts[0] + parts[1]
    R = 0.5 * (R + R.T)
    # R must be positive definite for (H P H^T + R)^-1: floor its eigenvalues
    eigvals, eigvecs = np.linalg.eigh(R)
    floor = 1e-6 * max(float(np.mean(np.diag(R))), 1e-30)
    R = (eigvecs * np.clip(eigvals, floor, None)) @ eigvecs.T
    return R, (f"{config['noise_scale']:g} x {config['r_noise_file']} + "
               f"{config['aliasing_scale']:g} x {config['r_aliasing_file']}")


def pseudo_open_loop(da):
    """The turbulence, as the controller rebuilds it: input / og - applied, og the WFS's optical
    gain (the controller's param, 1: rama_cup's pol). From rama_cup's pol = input - applied:
    pol + (1 / og - 1) input. After an og change, only the frames run with the new og."""
    telemetry = da.telemetry.current if "og" in getattr(da, "params_changes", []) else da.telemetry
    og = float(da.params.get("og", 1.0))
    return telemetry.pol + (1.0 / og - 1.0) * telemetry.input


MAX_LIMITED_PCT = 1.0      # no fit when RAMA's DM limiter changed more of the frames than this (%)


def limited_pct(telemetry):
    """The share (%) of the telemetry's frames whose command RAMA's DM limiter clipped or scaled
    down: their pseudo open loop is wrong (the DM did not apply what the controller asked for)."""
    return 100.0 * float(telemetry.dm_limited.mean()) if len(telemetry.seq) else 0.0


def unpack(params, n):
    """varn_lqg's params (flat lists) -> var [n, N n], K [N n, n], C [n, N n]."""
    size = len(params["var"]) // n
    shape = {"var": (n, size), "kalman": (size, n), "predict": (n, size)}
    return [np.reshape(np.asarray(params[k], float), shape[k]) for k in ("var", "kalman", "predict")]


def pack(var, K, C):
    """The other way: the params to send."""
    return {"var": var.ravel().tolist(), "kalman": K.ravel().tolist(), "predict": C.ravel().tolist()}


class VarNLQGFit(ramacup.Optimizer):
    def setup(self, da):
        da.config = {"telemetry_depth": 10000, "optimize_every": 2000,
                     # R = noise_scale * R_noise + aliasing_scale * R_aliasing (make_R.py)
                     "r_noise_file": "R_noise.npy", "r_aliasing_file": "R_aliasing.npy",
                     "noise_scale": 1.0, "aliasing_scale": 1.0,
                     # only without the R files: R = sigma_noise2 I, as var_lqg
                     "sigma_noise2": 0.3 / da.n_modes,
                     "dare_n_iter": 500,
                     # the learnt predictor (learned_predictor.py), from the second fit on:
                     # "all" (var, kalman, predict trained), "gain" (kalman, predict), "off"
                     "learn": "all",
                     "learn_seconds": 15.0,         # wall-clock budget of one training
                     "learn_batch": 512,            # frames per Adam iteration
                     "learn_lr": 3e-3,
                     "learn_val_frac": 0.3,         # held out: early stopping and the choice
                     "min_gain": 0.005,             # sent only if 0.5 % better than in place (held out)
                     # robust margin: the learnt predictor is moved back towards its start until
                     # the loop it closes through pol stays stable (spectral radius below
                     # robust_max_radius) for the WFS's optical gain off by these factors
                     "robust_og": [0.7, 1.0, 1.3],
                     "robust_max_radius": 1.0,
                     # each new model is tried for that many frames and undone if the residual
                     # got worse, at once if the loop diverges (ramacup's trial; 0: off)
                     "trial_frames": 1000}
        if da.config["learn"] not in ("all", "gain", "off"):
            raise SystemExit(f"optimizer_varn_lqg: learn {da.config['learn']!r}: 'all', 'gain' or 'off'")
        if da.config["learn"] != "off" and learned_predictor is None:
            print(f"optimizer_varn_lqg: WARNING no learnt predictor ({LEARN_ERROR}): the model only",
                  flush=True)
        self.fits = 0
        self.R_input, why = load_R(da.n_modes, da.config)
        if self.R_input is None:
            print(f"optimizer_varn_lqg: WARNING {why}: R = {da.config['sigma_noise2']:.3g} I "
                  f"(run make_R.py to make the R files)", flush=True)
        else:
            print(f"optimizer_varn_lqg: R = {why}, mean variance {np.mean(np.diag(self.R_input)):.3g}",
                  flush=True)

    def measurement_noise(self, da):
        """R in the units of the pseudo open loop, input / og: R_input / og^2."""
        n = da.n_modes
        if self.R_input is None:
            return da.config["sigma_noise2"] * np.eye(n)
        og = float(da.params.get("og", 1.0))
        return self.R_input / og ** 2

    def optimize(self, da):
        limited = limited_pct(da.telemetry)
        if limited > MAX_LIMITED_PCT:                       # the DM did not apply the outputs: no fit on them
            print(f"optimizer_varn_lqg: the DM limited {limited:.1f} % of the frames: no fit", flush=True)
            return
        n = da.n_modes
        order = len(da.params["var"]) // (n * n)            # the controller's VAR order
        pol = pseudo_open_loop(da).T.astype(float)          # (n_modes, frames)
        if pol.shape[1] <= 2 * order * n:
            print(f"optimizer_varn_lqg: {pol.shape[1]} frames for {order * n} coefficients per mode: "
                  f"no fit (raise telemetry_depth)", flush=True)
            return
        var, noise = fit_var(pol, order)
        A = companion(var)                                  # the state [pol_t, ..., pol_t-N+1]
        Q = np.zeros_like(A)
        Q[:n, :n] = noise
        R = self.measurement_noise(da)
        K, iterations = kalman_gain(A, Q, R, da.config["dare_n_iter"])
        delay = max(1.0, getattr(da, "delay_exact", da.delay))   # frames, may be fractional
        predict = ahead(A, delay)[:n]
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(predict))):
            print("optimizer_varn_lqg: non-finite Kalman gain or prediction: not sent", flush=True)
            return
        radius = np.max(np.abs(np.linalg.eigvals(A)))
        what = (f"{pol.shape[1]} frames -> VAR({order}) and Kalman predictor of {n} modes, |eig A| max "
                f"{radius:.4f}, Riccati {iterations} it., delay {delay:g}, og {da.params.get('og', 1.0):g}")
        self.fits += 1
        if da.config["learn"] == "off" or learned_predictor is None or self.fits == 1:
            # the model as it is: always without learning, and the first fit (the loop corrects
            # at once; the learning starts at the next fit, with more frames)
            da.set_params(pack(var, K, predict))
            print(f"optimizer_varn_lqg: {what}", flush=True)
            return
        self.learn(da, pol.T, (var, K, predict), delay, what)

    def learn(self, da, pol, model, delay, what):
        """
        Train the predictor on pol [frames, n] from the better of the model and the predictor in
        place (held out), and send the result if it beats the predictor in place by min_gain.
        """
        c = da.config
        current = unpack(da.params, da.n_modes)
        val = learned_predictor.held_out(pol, delay, c["learn_val_frac"])
        current_rms = learned_predictor.residual_rms(val, *current, delay)
        model_rms = learned_predictor.residual_rms(val, *model, delay)
        start, start_name = (model, "model") if model_rms < current_rms else (current, "in place")
        try:
            *learnt, info = learned_predictor.train_predictor(
                pol, *start, delay, train_var=(c["learn"] == "all"), val_frac=c["learn_val_frac"],
                batch=c["learn_batch"], lr=c["learn_lr"], max_seconds=c["learn_seconds"])
        except ValueError as error:                          # too few frames for the taps
            print(f"optimizer_varn_lqg: no learning ({error})", flush=True)
            learnt, info = start, {"held_out_best": min(model_rms, current_rms), "iterations": 0,
                                   "seconds": 0.0}
        # the margin: learning for the residual alone pushes the predictor to the edge of
        # stability; an optical gain that is not exactly og (a pyramid's moves with the residual)
        # then makes the loop diverge. Step back towards the start until it is robust.
        learnt, alpha, worst = learned_predictor.robust_mix(
            start, learnt, delay, rhos=c["robust_og"], max_radius=c["robust_max_radius"])
        best_rms = learned_predictor.residual_rms(val, *learnt, delay)
        summary = (f"held-out residual: in place {current_rms:.4g}, model {model_rms:.4g}, learnt "
                   f"{best_rms:.4g} (from {start_name}, {info['iterations']} it., {info['seconds']:.1f} s; "
                   f"robust step {alpha:.2f}, worst radius {worst:.3f})")
        if best_rms < (1.0 - c["min_gain"]) * current_rms:
            da.set_params(pack(*learnt))
            print(f"optimizer_varn_lqg: {what}; {summary}: sent", flush=True)
        else:
            print(f"optimizer_varn_lqg: {what}; {summary}: not better than in place, not sent", flush=True)


if __name__ == "__main__":
    da = ramacup.DataAccessOptimizer(name="varn_lqg")
    ramacup.run(da, VarNLQGFit())
    da.close()
