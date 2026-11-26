# varn_lqg: what is improved, and why

`varn_lqg` (in `rama_controller/varn_lqg/`) starts from the reference controller `var_lqg`. Each
frame it does the same thing: a Kalman filter on a vector autoregressive (VAR) model of all the
modes predicts the turbulence `delay` frames ahead, and the controller outputs its opposite. A
separate optimizer fits the model from the telemetry while the loop runs.

Three things are different.

## 1. The order N of the model is a choice

`var_lqg` always uses VAR(2): each prediction uses the last 2 frames. `varn_lqg` uses the last N
frames, with N set in the manifest (`--order N`, default 2).

**Why:** a longer memory can describe the turbulence better (for example slow oscillations, or
noise that partly averages out). In simulation, going from N = 2 to N = 4 lowered the residual
from 0.076 to 0.068 (8 modes, delay 2).

**Cost:** each step of the model has more numbers to fit (N × n_modes per mode), so it needs more
frames of telemetry. Sending the model to the controller also takes longer.

## 2. R comes from the WFS, not from a guess

In a Kalman filter, R says how noisy each measurement is: how much to trust the WFS against the
model's own prediction. `var_lqg` assumes `R = 0.3 / n_modes` for every mode. `varn_lqg` uses
`R = R_noise + R_aliasing`:

- **`R_noise`**: photon and read-out noise of the pyramid, propagated through the reconstructor.
- **`R_aliasing`**: the high-order turbulence the DM cannot correct, which the WFS sees as
  low-order modes.

Both are computed once, offline, by `make_R.py` with AI4AO, and saved as `.npy` files in the
folder.

**Why:** with the wrong R, the filter trusts the measurements too much or too little. In
simulation (8 modes, VAR(2)):

| R used | residual, delay 2 | residual, delay 2.3 |
|---|---|---|
| the right R | 0.076 | 0.081 |
| `var_lqg`'s 0.3/n | 0.096 | 0.100 |
| integrator, for reference | 0.094 | 0.100 |

With the guessed R, the Kalman controller was no better than the integrator.

**Limits:**
- **Conditions:** R depends on them. The noise depends on the flux (`--nphotons`; the WFS
  model's default, 10⁷ photons, makes it negligible), and the aliasing on r0, roughly as
  r0^(−5/3). The params file's r0 range (0.02–0.2 m) averages over a factor of ~50. Make the
  files for the conditions of the Run (`--r0`), or adjust them with the optimizer's
  `noise_scale` and `aliasing_scale`.
- **Units:** R must be in the units of the controller's input. `make_R.py` uses AI4AO's linear
  reconstructor on RAMA's `M2C` (96 modes), so RAMA's modes must be the same basis. With the CNN
  input, a linear noise propagation only approximates the CNN's noise.
- **Optical gain:** the controller works on `input / og`, so the optimizer divides R by `og²`.
- **Double counting:** `make_R.py` switches the WFS noise off while it computes the aliasing.
  Otherwise the noise would be counted twice. Your notebook's version left it on, though at the
  default 10⁷ photons its effect was negligible.

## 3. The predictor is learnt on the telemetry (the main improvement)

The model-based way builds the predictor in steps: fit A by least squares, take Q from the fit
residuals, assume R, solve the Riccati equation. Each step has an error:

- least squares is **biased by the WFS noise**, so the model thinks the turbulence changes faster
  than it does;
- `Q` and `R` are **estimates**, and the gain is only as good as they are.

The learnt predictor keeps the same structure: the same three matrices `var`, `kalman`,
`predict`, the same controller, the same cost per frame. But it trains them directly to make the
residual as small as possible on the recorded telemetry. That residual is what the controller
actually leaves, and the competition's score is based on it.

**Why it works:**
- **It aims at the score.** It minimises the residual itself, not a model fit.
- **The noise doesn't bias it.** The WFS noise is white, so nothing can predict it. It only adds
  a constant to the residual and doesn't move the best matrices.
- **The R and Q errors fade away.** Learning starts from the model-based predictor (with your
  R), and whatever the start, it ends at about the same place. R becomes a starting point, not
  something that has to be exactly right.

### Least squares also uses the telemetry: what is the difference?

Both use the same frames. They differ in **what they minimise** and **which matrices that
minimisation actually sets**.

**Least squares** fits only `var`, by predicting the next raw measurement from the previous raw
measurements:

```
var = argmin  Σ_t ‖ pol_{t+1} − var · [pol_t, pol_{t−1}, …] ‖²
```

Everything else is derived from it, not fitted to the data:

```
Q       = cov(residuals of that fit)
R       = R_noise + R_aliasing                (make_R.py)
kalman  = Riccati(A, Q, R)                    optimal only if A, Q and R are exactly right
predict = (A^delay)[:n]                       A's error compounds over the delay
```

This has two problems:

1. **The WFS noise biases `var`.** Least squares treats the past measurements as exact, but they
   are noisy. For one mode with AR(1) turbulence `φ_{t+1} = a φ_t + w` measured as
   `pol = φ + v`:

   ```
   a_LS = a · σ_φ² / (σ_φ² + σ_v²)
   ```

   With `a = 0.95` and noise as large as the signal, least squares gives 0.475: the turbulence
   looks twice as fast as it is. On an AR(2) test (true 1.60 / −0.65) it gave 0.54 / 0.30.
   The filter's R does not undo this: R only says how much to trust a measurement, and `var` is
   already wrong.
2. **It solves a different task.** It scores raw measurements predicting the next measurement.
   The controller filters the measurements through `kalman`, then predicts `delay` frames ahead
   with `predict` (blended for a fractional delay). Least squares never sees `kalman`, `predict`
   or the delay.

**The learnt predictor** writes the whole controller as one function of the three matrices, runs
it on the telemetry, and measures the residual it would have left:

```
x'_k  = A x_{k−1}                                   A = [[var], [I 0]]
x_k   = x'_k + kalman (pol_k − H x'_k)              the filter
p_k   = predict · x_k                               the prediction
res_t = pol_t − (1−f) p_{t−n} − f p_{t−n−1}         what is left (delay = n + f)

(var, kalman, predict) = argmin  Σ_t ‖ res_t ‖²
```

**Why the noise does not bias it.** The target `pol_t = φ_t + v_t` is noisy, but `v_t` is
independent of the earlier frames the prediction uses. The error splits in two:

```
E‖res_t‖² = E‖φ_t − prediction‖² + σ_v²
```

The second term is a constant that no matrix can change. So minimising the noisy residual is the
same as minimising the true one: it finds the best predictor of the real turbulence from the
noisy past.

| | least squares (model-based) | learnt |
|---|---|---|
| fitted to the data | `var` only | `var`, `kalman`, `predict` together |
| objective | next raw measurement from the previous ones | the residual after the full filter and the `delay`-step prediction |
| WFS noise | biases `var` (dynamics too fast) | adds a constant only, no bias |
| `kalman` | Riccati: needs the right Q and R | fitted to the data |
| `predict` | `A^delay`: compounds A's error | fitted to the data |
| fractional delay | only through `predict` | inside the objective |

The two give the same result when there is no noise and the order is right: then least squares
finds the true `var`, the Riccati gain is optimal, and learning has nothing to improve.

**The learnt matrices are no longer physical.** After learning, `var` is not "the turbulence
dynamics" and `kalman` does not solve a Riccati equation. Only their combination matters. The
command is a linear filter on the past measurements,
`p_k = Σ_j predict · F^j · kalman · pol_{k−j}` with `F = (I − kalman H) A`, and many sets of
matrices give the same filter. That is fine for the controller, but do not read physics into
the learnt `var`.

**Results in simulation** (8 modes, a fit every 1000 frames on the last 2000):

| delay | integrator | model-based (`varn_lqg`) | learnt | best possible linear predictor |
|---|---|---|---|---|
| 2.0 | 0.094 | 0.075 | **0.065** | 0.062 |
| 2.3 | 0.101 | 0.079 | **0.065** | 0.064 |

Offline, starting from an R wrong by a factor 10 (either way) or from `var_lqg`'s guess, the
model-based predictor got up to 35 % worse. The learnt one ended at nearly the same value
whatever its start: on 8 modes, 1-4 % above the best possible. There was one exception: on 32
modes with 2000 frames, starting from R / 10, it found nothing better and kept the start.

With 96 modes (RAMA's size) there are more numbers to learn, so it needs more data:

| telemetry | model-based | learnt | best possible |
|---|---|---|---|
| 5000 frames | 0.084 | 0.077 (−7 %) | 0.066 |
| 10 000 frames | 0.082 | 0.073 (−11 %) | 0.066 |

**A full 60 s Run** at 500 Hz (30 000 frames, delay 2.3, VAR(2), the optimizer's defaults). The
"whole run" counts the learning time, as the score does:

| | 8 modes, whole run | 8 modes, last third | 96 modes, whole run | 96 modes, last third |
|---|---|---|---|---|
| integrator | 0.099 | 0.099 | 0.102 | 0.102 |
| model-based (`learn` off) | 0.082 | 0.078 | 0.086 | 0.082 |
| learnt (`learn` "all") | **0.070** | **0.064** | **0.079** | **0.072** |
| best possible linear predictor | 0.064 | 0.064 | 0.066 | 0.066 |

Over the whole Run, learning lowered the residual by 14 % (8 modes) and 8 % (96 modes) compared
with the model-based predictor. Over the last third it was 18 % and 12 %. Few predictors were
actually sent (2 of 15 fits on 8 modes, 5 of 15 on 96), so the cost in missed frames stays
small.

**How it runs during a Run:**
1. **First fit** (after 2000 frames): the model-based predictor is sent at once, so the loop
   starts correcting early.
2. **Every 2000 frames after that:** the optimizer fits the model again. It then trains the
   predictor, starting from whichever is better: the new model or the predictor already in place.
   Training continues from fit to fit, on up to the last 10 000 frames.
3. **Safety:**
   - **Held-out check:** the last 30 % of the telemetry is held out. The result is judged on
     those frames, not on the ones it trained on.
   - **When it sends:** a new predictor is sent only if it is at least 0.5 % better there,
     since every send costs a few missed frames.
   - **Stability:** a filter that would be unstable is refused, and the robust step keeps a
     margin for an optical gain 30 % off (see below).
   - **Time limit:** each training stops after 15 s at most. The loop never waits for it.
4. **Off switch:** config `learn`: `"all"` (default), `"gain"` (train only `kalman` and
   `predict`), or `"off"` (the model-based predictor alone). Without torch it falls back to
   `"off"` by itself.

### The robust step: why it is needed

The first version of the learnt predictor diverged in the AI4AO simulation (`SimpleKalman`'s:
pyramid with modulation 2, 80 modes), within about 20 frames. The cause was the WFS's
**optical gain**: the pyramid sees only `og ≈ 0.6–0.85` of the residual, and less when the
residual is large. The pseudo open loop `z + u_{k−d}` then holds part of the controller's own
output, `og·φ + (1 − og)·u_{k−d}`, which closes a hidden loop:

```
x_k = F x_{k−1} + K (1 − ρ) C x_{k−d}        ρ = true og / og the controller uses
```

Training for the residual alone pushes the predictor to the edge of this loop's stability. Two
fixes, both now in `varn_lqg` and in the tutorial notebook:

1. **Compensate the optical gain**: `pol = z / og + u_{k−d}`. `varn_lqg` already has the `og`
   param. Set it on the bench.
2. **The robust step** (`robust_mix`): move the learnt predictor back towards its start,
   `(1 − α)·start + α·learnt`, to the largest α whose hidden loop stays stable for `og` off by
   ×0.7 to ×1.3. In the synthetic tests (og = 1) it kept everything (α = 1). In the AI4AO
   simulation it kept 0.86–0.94 of the step.

AI4AO simulation, Strehl over 2000 frames:

| run | model-based | learnt, no margin | learnt, robust |
|---|---|---|---|
| og compensated (tutorial notebook) | 40.2 % | 58.3 % | **64.1 %** |
| og compensated (diagnostic) | 39.0 % | 58.4 % | **59.2 %** |
| og = 1, not compensated (diagnostic) | 28.7 % | diverges | **48.3 %** |

The numbers move a lot between runs (different telemetry and turbulence), so compare within a
row.

### How the training works

- **Optimiser:** Adam on the three matrices, from the starting predictor, in torch on the CPU
  with one thread (the optimizer runs at low priority, so the loop keeps its CPU).
- **The speed trick.** Running the filter frame by frame inside the gradient is slow: about 40 s
  per training on 8 modes. The filter is linear and time-invariant, so the prediction is also a
  sum over past frames, `p_k = Σ_j G_j pol_{k−j}` with `G_j = predict · F^j · kalman`. The
  training computes `G_j` once per iteration, truncated when `F^j` becomes negligible (50 to 400
  frames), and scores it on a random batch of 512 frames in one product. That brought it to
  about 3 s on 8 modes, with the same result.
- **Exact checks.** Early stopping and the final choice use the exact frame-by-frame filter on
  the held-out frames, so the truncation cannot hide an error.
- **Stopping:** when the held-out residual has not improved for 20 checks (one check every 25
  iterations), or after `learn_seconds`.
- **Scaling:** the data are divided by their standard deviation before training. The predictor
  is linear, so this changes nothing in the result but keeps the numbers well conditioned.

## What it costs

| | value |
|---|---|
| controller, per frame (96 modes, numpy, one thread) | ~13 µs (N = 1), ~20 µs (N = 2), ~26 µs (N = 4), ~40 µs (N = 6), against 2 ms per frame at 500 Hz |
| params sent to the controller | `3 N n_modes²` numbers: 55 000 for N = 2, 110 000 for N = 4 (96 modes) |
| one send | about ten missed frames per 55 000 numbers, with the present rama_cup server (rama_controller README) |
| model-based fit (96 modes, VAR(2)) | well under a second |
| one training (96 modes) | up to `learn_seconds` (15 s). An iteration on 5000 frames takes 0.15 s on all of them, ~50 ms with minibatches (checks included) |
| first fit | after 2000 frames (4 s at 500 Hz); the loop outputs zero until then, as in `var_lqg` |

The defaults changed from `var_lqg`. Fits happen every 2000 frames (was 1000), on up to the last
10 000 frames (was 1000). More frames help both least squares and learning.

## Other ways to remove the noise bias

The learnt predictor is one way. Two others were considered and could be combined with it:

- **Instrumental variables.** Fit `var` against older frames, which share no noise with the
  regressors. On the AR(2) test this recovered 1.57 / −0.63 (true 1.60 / −0.65). For VAR(N),
  the instrument must be N frames older. `estimate_A(..., "instrumental_variable")` in
  `kalman/Kalman_tools*.py` uses only 1, so it stays biased for N ≥ 2 (0.75 / 0.14 on the same
  test): a bug to fix there.
- **Noise-corrected Q:** `Q = Q_raw − R − A R Aᵀ`, from `kalman/`.

Both still go through the Riccati equation, so they still rely on the right R. The learnt
predictor does not.

## Kept from var_lqg

These are unchanged:
- the pseudo open loop and the optical gain `og`;
- the fractional loop delay (e.g. 3.3 frames);
- the warm start when a new model arrives;
- the reset when the output becomes non-finite;
- no fits on frames the DM limiter changed;
- rama_cup's trial, which undoes a model that makes things worse.

## What is not known yet

- **Every number above is from a simulation.** Its turbulence is exactly a linear random process,
  which is the best case for this method. The twin and the bench will be less ideal.
- **The R files have not been made yet.** Run `make_R.py` with the bench's flux and r0.
- **Not yet run under the real rama_cup server.** That is the next test: `ramaCupSim`, then the
  twin.
- **Training time was measured on this laptop.** The bench's CPU and load will change it.
- **On 96 modes it is still far from the best possible** (0.072 against 0.066 at the end of
  a Run). More telemetry, or `learn` "gain" (fewer numbers to fit), may help. Both are untested.
- **rama_cup's behaviour with two quick sends:** with learning on, each fit sends at most once.
  How rama_cup's trial reacts to a send during a running trial is not known.
- **The `--order` argument** in the manifest assumes `ramacup` does not parse the command line
  itself.
- **`install_deps.sh` must be executable** on the Linux machine (`chmod +x`). Windows may not
  keep the bit.

## Files

| file | what it is |
|---|---|
| `rama_controller/varn_lqg/controller_varn_lqg.py` | the controller (real time) |
| `rama_controller/varn_lqg/optimizer_varn_lqg.py` | the optimizer: model-based fit, then learning |
| `rama_controller/varn_lqg/varn_model.py` | the model-based part: least squares, Riccati, `A^delay` (numpy) |
| `rama_controller/varn_lqg/learned_predictor.py` | the training (torch) |
| `rama_controller/varn_lqg/make_R.py` | offline: `R_noise.npy`, `R_aliasing.npy`, `R_info.json` (needs AI4AO) |
| `rama_controller/varn_lqg/manifest_varn_lqg.yaml`, `install_deps.sh`, `README.md` | rama_cup's files |
| `experiments/compare_learned.py` | the comparison in this document (outside the submission) |
| `../Notebooks/VarNLQGTutorial.ipynb` | step by step, on `SimpleKalman`'s AI4AO simulation |

To reproduce the numbers (about 20 minutes on one core, no RAMA needed):

```bash
cd RamaCup/experiments
python compare_learned.py
```

It runs `varn_lqg`'s own controller and optimizer against a simulated loop with a stand-in for
`ramacup`. The simulated turbulence is a VAR(2) process on 8, 32 or 96 correlated modes, with
white WFS noise. Its true model gives the "best possible linear predictor" of the tables.

## Next steps

1. **Make the R files** with `make_R.py`, for the bench's flux and r0.
2. **Run on rama_cup's simulated loop** (`ramaCupSim varn_lqg`): the real server, params
   channel and timing. Compare it with `var_lqg`, `lqg` and the integrator.
3. **On the twin**, with the CNN input then the linear one, and with `og` below 1.
4. **Tune for 96 modes:** a larger `telemetry_depth`, `learn` "gain", and N = 3–4. Watch the
   missed frames.
5. **Step 2 of the plan, only if step 1 holds up on the bench:** a small neural-network
   correction on top of the linear predictor, `f(x) = A x + g(x)`. It is worth it only where the
   data are not linear (the pyramid's optical gain, the DM limiter).
