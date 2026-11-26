# RAMA Cup controller: assessment and plan

How our Kalman work (`kalman/`, `Notebooks/DAOKalman.ipynb`, `Notebooks/DAOAdaptiveKalman.ipynb`,
`REALTIME_PLAN.md`) compares with the RAMA Cup framework and its reference controllers
(`rama_controller/`), and how to turn it into an entry.

## 1. What the competition expects

- **Two programs per entry.** `controller_<name>.py` runs `step(da)` every frame: `da.input`
  (96 modes, float32) → `da.output`. `optimizer_<name>.py` reads `da.telemetry`
  (`input`, `output`, `pol`, `dm_limited`) whenever it likes and changes the controller's
  **params** with `da.set_params`. The `ramaCupServer` sits between them and RAMA. It enforces the
  deadline (one frame period), checks for NaN and wrong lengths, zeroes the output when the loop
  opens, and records everything.
- **One folder per entry, self-contained.** The folder holds a manifest, `install_deps.sh` (a conda
  env named after the controller) and everything the code reads. Nothing is installed outside it,
  there are no `../` paths, and it is **CPU only**.
- **A Run is one scored window.** It lasts 60 s on the bench (500 Hz, about 30 000 frames) and 180 s
  on the twin. Learning time counts in the score (rms of the residual modes). Entries are also
  judged on Strehl, stability, efficiency and originality.
- **Loop conditions set by the server.** The delay is given as `da.delay_exact` and can be
  fractional (3.3 on the bench). The WFS optical gain `og` is below 1 on the bench. The DM limiter
  marks frames whose `pol` is wrong. With the CNN input (the default), modes 80-95 are exactly 0.

## 2. Assessment: how our controller compares

### Same as `var_lqg` (the closest reference)

`var_lqg` is essentially the controller we already have:

| | ours (DAOKalman) | `var_lqg` |
|---|---|---|
| state | `[x_k, x_{k-1}]`, AR(2) on all modes jointly | same (VAR(2)) |
| pseudo open loop | `y_k = z_k + u_{k-d}` | `pol = input/og - applied` (same up to `og`) |
| filter | steady-state `K` (after your fix) | steady-state `K` (iterated Riccati) |
| command | `-(A^d x)[:m]` | `-(predict @ x)`, `predict = A^d[:n]` |
| fit vs loop | fit in a separate process (REALTIME_PLAN P2) | fit in the optimizer process |

So the core algorithm is **not** different. Entering it as it stands would score about like
`var_lqg`. On the simple simulated loop, `var_lqg` gets 0.116 against 0.115 for the integrator
(rama_controller README).

### Different, and worth keeping (our edge)

1. **Fit that is not biased by noise.** `var_lqg` fits the VAR by least squares on `pol`, which
   contains the WFS noise, so the fitted dynamics come out too damped. We have the
   instrumental-variable estimator and the noise-corrected `Q = Q_raw - R - A R A^T`
   (`estimate_A`, `estimate_Q`).
   **Bug to fix first:** for AR(p) on the augmented state, the instrument must be lagged by p
   augmented steps. `estimate_A(..., "instrumental_variable")` uses `x_{k-1}`, which shares
   `pol_{k-1}` with the regressor. Synthetic AR(2) test with true coefficients 1.60 and -0.65 at an
   SNR of about 2:

   | estimator | a1 | a2 |
   |---|---|---|
   | least squares (what `var_lqg` does) | +0.54 | +0.30 |
   | current instrumental variable (instrument `x_{k-1}`) | +0.75 | +0.14 |
   | instrumental variable with instrument `x_{k-2}` | **+1.57** | **-0.63** |

2. **A real `R`.** `var_lqg` uses `R = 0.3/n · I`. We have a physical noise model (photon +
   readout noise through the reconstructor) plus aliasing (`R_noise + R_aliasing`), and in
   `controller.py` an `R_fraction` option.
3. **Optical gain measured, not guessed.** `DAOAdaptiveKalman` measures the WFS gain per mode with
   a dither and adds a margin. In `var_lqg`, `og` is a manual param.
4. **Health checks.** `controller.py` has NIS monitoring with the whitened innovation, a fallback
   to the integrator, and recovery only with a new model. The references only have a non-finite
   guard and rama_cup's trial/undo.
5. **Integrator while learning.** We identify on integrator telemetry, so the loop corrects while
   the model is being learnt. `var_lqg` outputs zero until its first fit (open loop), and the
   score counts that time.

### Missing in ours, needed for the competition

| needed | ours now | what to do |
|---|---|---|
| fractional delay (`delay_exact`, 3.3) | integer `delay = 2` | blend `(1-f) u_{k-n} + f u_{k-n-1}` in `pol`, and predict with `(1-f) A^n + f A^{n+1}` (copy `split_delay`, `applied`, `ahead` from `var_lqg`) |
| 96 RAMA modes, CNN modes 80-95 = 0 | 70 modes of the AI4AO M2C basis | Kalman on the first `n_kf` modes (config: 80 for the CNN input, 96 for linear), integrator on the rest |
| CPU only, no AI4AO at run time | torch/CUDA, AI4AO `PyramidWFS` | numpy only. Estimate `R` from data, or ship a precomputed `.npy` **only if** it is in RAMA's mode basis and for the right reconstructor (a linear noise propagation does not describe the CNN) |
| params channel (fixed names, types, list lengths) | DAO SHMs and a deque | the model goes through `da.set_params` as flat lists, with sizes fixed in `setup` |
| `da.reset`, warm start on a new model, non-finite guard | none | copy `var_lqg`'s pattern (replay the last 50 `pol` frames through a new model) |
| DM limiter | not handled | skip fits when more than 1 % of the frames have `dm_limited` set |

### What to drop for the competition

- **The custom DAO SHM layout** (`REALTIME_PLAN.md` sections 4-5). rama_cup already provides those
  pieces. Our processes map onto it like this:

  | ours | rama_cup |
  |---|---|
  | P0, real-time controller | `controller_<name>.py` |
  | P2, identification and gain | `optimizer_<name>.py` |
  | P1, logger | the server's recording and the Telemetry |
  | P3, supervisor | the server's safety rules and the trial in the optimizer |
  | MODEL SHM double buffer | params, applied atomically between frames |

  Keep REALTIME_PLAN for the direct-DAO setup outside the cup.
- **The split critical path** (`u = b + G z`, then update after sending). In rama_cup, `step` has
  to return before the output leaves, and the round trip through the server (60-240 µs) dwarfs
  the about 10 µs of compute. The fixed-gain step you have now is enough. The hard constraint is
  the frame period (2 ms at 500 Hz), which the old 1.5 ms per step nearly broke.

## 3. Plan

### 3.1 Folder (the name is your choice; `<name>` below)

```
RamaCup/<name>/                 added to RAMA_CUP_CONTROLLERS, next to rama_controller/
├── manifest_<name>.yaml        controller + optimizer, as var_lqg's
├── install_deps.sh             ramaCupEnv create <name>   (numpy only: rama's env is enough)
├── controller_<name>.py        real time
├── optimizer_<name>.py         identification + gain + validation
├── kalman_core.py              pure-numpy functions, copied from kalman/Kalman_tools_numpy.py
│                               (the folder may not import from ../kalman)
├── tests/test_kalman_core.py   synthetic, fixed seed, no ramacup needed
└── README.md                   as the references: what it does, params, config, how to run
```

### 3.2 Controller (`controller_<name>.py`)

- **Params** (sizes fixed in `setup`, so the AR order is fixed too: allocate `p_max` and pad with
  zeros if model selection may pick a lower order):

  | param | size (n = `n_kf`) | meaning |
  |---|---|---|
  | `var` | n × 2n | top rows of `A` |
  | `kalman` | 2n × n | steady-state `K` |
  | `predict` | n × 2n | `ahead(A, delay_exact)[:n]` |
  | `og` | 1 | optical gain (manual override, or set by the optimizer) |
  | `use_kalman` | bool | false = integrator on all modes (bootstrap and fallback) |
  | `gain`, `leak` | float | integrator, also used for modes ≥ `n_kf` |

- **Step:**
  1. `pol = input/og - applied()`, with the fractional delay.
  2. Fixed-gain update `x = A x + K (pol - H A x)` (exploit `A = [[var], [I 0]]`: no
     2n × 2n product).
  3. `u = -predict @ x` on the first `n_kf` modes, and the integrator on the rest.
  4. Non-finite or `|u| > U_MAX` → reset the state, output 0 for that frame, set
     `use_kalman = false` locally.
  5. Preallocate the buffers, with no `concatenate` in `step`.
- On `params_changed`, rebuild `A` from `var` and warm-start on the last 50 `pol` frames before
  acting. On `da.reset`, zero the state and the past outputs.
- While `use_kalman` is false the output is the integrator. `pol` stays valid under any
  controller, so the optimizer learns from integrator telemetry while the loop is corrected.

### 3.3 Optimizer (`optimizer_<name>.py`, 1 thread, `os.nice(10)`, as the references)

1. **First fit early:** `telemetry_depth` about 2000-3000 frames (4-6 s at 500 Hz), then refit
   with more data. Skip fits on DM-limited frames.
2. **Optical gain:** use the `og` param by default. Optional: a short dither during the bootstrap.
   This needs a `dither_amp` param in the controller, with the optimizer estimating
   `og = corr(input, dither at lag delay)`, as in `DAOAdaptiveKalman` cell 5, with the margin
   (overestimating is the safe side).
3. **Identification on `pol`:** remove the mean, then fit `A` by instrumental variable with
   instrument `x_{k-p}`. Fall back to least squares + ridge if `Z^T X` is ill-conditioned (short
   data, the zero CNN modes). Then `Q = Q_raw - R - A R A^T`, projected onto PSD, and only the
   top-left block is nonzero.
4. **`R`, in order of preference:**
   - from data, by the MA structure of the one-step residual (its lag-1 covariance holds `R`), or
     by the high-frequency floor of `pol`'s spectrum per mode;
   - then a single scalar `r_scale` on top, tuned by the optimizer like `gain_search` tunes the
     gain, using rama_cup's `trial_frames` to undo a worse value.
5. **Gain:** steady-state DARE. Port `steady_state_gain` (doubling) from `kalman/Kalman_tools.py`
   to numpy, which handles the singular AR `Q`. Then `predict = ahead(A, delay_exact)[:n]`.
6. **Validate before sending:**
   - `A` and `F = (I-KH)A` stable, and everything finite;
   - the `delay`-step prediction error on the held-out last 30 % of `pol` lower than the current
     model's and than persistence (`pol_{k+d} ≈ pol_k`, which is what the integrator achieves).
     This is the offline proxy for the residual, and it also chooses the AR order, ridge and
     `r_scale` without risking the loop.
7. **Send rarely:** about 55k numbers cost about 10 missed frames per send (rama_controller README),
   and misses count against stability. Refit on a schedule, but only call `set_params` when
   validation shows a clear gain. Reduce `n_kf` when the input is CNN (80, not 96).
8. Set `use_kalman = true` with the first validated model. rama_cup's trial (`trial_frames`)
   undoes a model that makes the residual worse.

### 3.4 Order of work

1. **Core and tests** (`kalman_core.py`, about 1 day):
   - IV with instrument `x_{k-p}`, with a test that recovers the AR coefficients under noise. Fix
     the same bug in `kalman/Kalman_tools*.py` and its tests.
   - Numpy DARE (doubling), with a test against iterating `kalman_filter`.
   - `split_delay` / `ahead` with a fractional delay.
   - Data-based `R`.
2. **Controller + optimizer on `ramaCupSim`:**
   - `--sim-delay 2`, then a fractional delay;
   - compare with `integrator`, `var_lqg` and `lqg` (the README table: 0.115 / 0.116 / 0.113 on
     24 modes);
   - check the misses and latency in the server's status line.
3. **Replay** of a recording (`RAMA_CUP_DA=replay:`): check the signs, delays and that the outputs
   are finite.
4. **Twin** (`ramaCupPipeline start --twin --atm`), with the CNN input then linear, and with
   `og < 1`.
5. **Campaign**: our entry against `var_lqg`, `integrator` and `omgi`, in sim then on the twin
   (`campaign_rama_controller.yaml` as a template). Look at `t_close` and the rms over the whole
   window.
6. **Optional:** the controller in C (`sdk/c`) with `--spin` for the lowest latency. The step is
   3 small GEMVs. Only worth it if latency shows up in the score.
7. Pack with `rama_cup_prep_controller.py`.

### 3.5 Open questions

- **CNN or linear input?** The aliasing / noise `R` from AI4AO only makes sense for one of them,
  and in RAMA's 96-mode basis.
- **Dither or manual `og`?** The dither costs a little residual during the bootstrap, while a
  wrong `og` destabilises predictors on the bench (rama_controller README).
- **`telemetry_depth` cap:** check how large the server allows `telemetry_depth` (more data =
  better VAR fit).
