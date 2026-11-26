"""VARN-LQG -- one Kalman predictor for all the modes together, on a VAR(N) model: the controller.

As var_lqg, with the order N of the model a choice (var_lqg: N = 2). Every frame:
  1. pol = input / og - output(delay frames ago)    the turbulence, without the DM (pseudo open loop)
  2. a single Kalman filter on the joint VAR(N) model of all the modes takes pol in (state: the
     last N pol vectors, N n_modes values), and predicts the turbulence `delay` frames ahead
  3. output = -prediction

The order: ORDER below, or `--order N` on the command line (the manifest's command). It fixes the
size of the params; the optimizer reads it from there.

The model is params, fitted by the optimizer optimizer_varn_lqg.py:
    "var"      (n_modes x N n_modes) pol_t+1 = var @ [pol_t, ..., pol_t-N+1] + noise
    "kalman"   (N n_modes x n_modes) the steady-state Kalman gain
    "predict"  (n_modes x N n_modes) the first n_modes rows of A^delay
Until the first fit they are zero: the output is zero (open loop, clean data for the fit).

A fractional loop delay (da.delay_exact, e.g. 3.3 = n + f) counts as the blend of the two
outputs around it, as in rama_cup's pseudo open loop: output(delay frames ago) is
(1 - f) output(k - n) + f output(k - n - 1).

    RAMA_CUP_DA=socket:ipc:///tmp/cup-varn_lqg.sock python3 controller_varn_lqg.py --order 3
"""
import argparse
import time
from collections import deque

import numpy as np
import ramacup

ORDER = 2               # N, the order of the VAR model; `--order N` overrides it
WARM = 50               # frames of pol a new model is run on before it acts (warm start)


def order_from_argv():
    """The VAR order: `--order N` on the command line, else ORDER."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--order", type=int, default=ORDER)
    order = parser.parse_known_args()[0].order
    if order < 1:
        raise SystemExit(f"controller_varn_lqg: --order {order}: must be at least 1")
    return order


def split_delay(delay):
    """The loop delay, frames, may be fractional (da.delay_exact, e.g. 3.3): n whole frames and f,
    the command reaching the WFS being (1 - f) output(k - n) + f output(k - n - 1), as in
    rama_cup's pseudo open loop."""
    delay = max(1.0, float(delay))
    n = int(delay + 1e-9)
    f = delay - n
    return n, (f if f > 1e-9 else 0.0)


def check_og(params):
    """A params change (the panel, the optimizer): og, the WFS's optical gain, in [0.05, 2]."""
    og = params.get("og", 1.0)
    return None if 0.05 <= og <= 2.0 else f"og {og}: must be between 0.05 and 2"


class VarNLQG(ramacup.Controller):
    def __init__(self, order=ORDER):
        self.order = order

    def setup(self, da):
        n, order = da.n_modes, self.order
        da.config = {"order": order}
        size = n * order * n
        da.params = {"var": [0.0] * size, "kalman": [0.0] * size, "predict": [0.0] * size, "og": 1.0}
        self.n_delay, self.f_delay = split_delay(getattr(da, "delay_exact", da.delay))
        self.past_outputs = deque(maxlen=self.n_delay + 1)   # newest last
        # warm start: a new model (a fit, an og change) is run on the last WARM frames of pol
        # before it acts, so that it starts in the state it would have now (pol is the turbulence,
        # whatever the controller did): no cold start, no dip in the correction at each fit
        self.pol_history = deque(maxlen=WARM)
        da.on_params(check_og)                          # og: refused outside [0.05, 2]
        self.last_warning = 0.0
        self.load_model(da)

    def load_model(self, da):
        self.og = float(da.params.get("og", 1.0))      # the WFS's optical gain: input = og x residual
        n, order = da.n_modes, self.order
        self.var = np.reshape(np.asarray(da.params["var"], dtype=float), (n, order * n))
        self.kalman = np.reshape(np.asarray(da.params["kalman"], dtype=float), (order * n, n))
        self.predict = np.reshape(np.asarray(da.params["predict"], dtype=float), (n, order * n))
        self.state = np.zeros(order * n)                 # [pol_t, ..., pol_t-N+1], estimated
        self.predicted_state = np.zeros(order * n)       # preallocated: no allocation per frame

    def applied(self):
        """What the DM applies now: (1 - f) output(k - n) + f output(k - n - 1), 0 before."""
        past, n, f = self.past_outputs, self.n_delay, self.f_delay
        out = (1.0 - f) * past[-n] if len(past) >= n else 0.0
        if f and len(past) >= n + 1:
            out = out + f * past[-n - 1]
        return out

    def update(self, pol):
        """The Kalman filter takes pol in; the prediction of the turbulence, delay frames ahead."""
        n = self.var.shape[0]
        predicted = self.predicted_state                 # x' = A x, A the VAR(N) companion matrix:
        np.dot(self.var, self.state, out=predicted[:n])  # the newest block from the model,
        predicted[n:] = self.state[:-n]                  # the others shifted by one frame
        innovation = pol - predicted[:n]
        np.dot(self.kalman, innovation, out=self.state)  # x = x' + K (pol - H x')
        self.state += predicted
        return self.predict @ self.state                 # the turbulence, delay frames ahead

    def restart(self, da, why):
        """The model's state and the past outputs from zero (its params kept), with a line at
        most once a second."""
        self.past_outputs.clear()
        self.pol_history.clear()
        self.state[:] = 0
        now = time.monotonic()
        if now - self.last_warning > 1.0:
            print(f"controller_varn_lqg: {why}: state reset (og {self.og:g}: a model unstable on this "
                  f"loop, the pseudo open loop wrong: try another og)", flush=True)
            self.last_warning = now

    def step(self, da):
        if da.reset:
            self.past_outputs.clear()
            self.pol_history.clear()
            self.state[:] = 0
        if da.params_changed:                            # a new model: warmed on the recent pol
            self.load_model(da)
            for past_pol in self.pol_history:
                self.update(past_pol)
        pol = da.input / self.og - self.applied()        # the turbulence (pseudo open loop)
        self.pol_history.append(pol)
        prediction = self.update(pol)

        output = -prediction
        if not np.all(np.isfinite(output)):              # the model blew up: start again from zero
            self.restart(da, "non-finite output")
            output = np.zeros(da.n_modes)
        da.output = output
        self.past_outputs.append(da.output)


if __name__ == "__main__":
    da = ramacup.DataAccessM2M(name="varn_lqg")
    ramacup.run(da, VarNLQG(order_from_argv()))
    da.close()
