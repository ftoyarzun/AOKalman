# How the learnt predictor is trained, step by step

The code is [`train_predictor`](rama_controller/varn_lqg/learned_predictor.py#L171) in
`rama_controller/varn_lqg/learned_predictor.py`.

## The short answer

You are mostly right. The training is **one optimization of `var`, `K` and `C` together, on all
the recorded telemetry, with no closed loop simulated.** Nothing is applied to a DM or a WFS,
and the turbulence does not depend on the candidate matrices: it is the recorded pseudo open loop
`pol`, fixed.

There is a recursion, but it is not a simulated loop. It is the **Kalman filter's own memory**.
The prediction at frame `k` comes from the state `x_k`, and the filter builds `x_k` from *all*
the past `pol`, using `var` and `K`. So the inputs of the regression are not the data
themselves, as they are for least squares. They are computed from the matrices being fitted.
That is what removes the closed form and forces gradient descent.

The rest of this file explains this, then goes through the steps.

## 1. Why least squares has a closed form here and the predictor does not

**Least squares for `var`.** The model is

    pol_t+1 = var @ s_t + e_t+1,      s_t = [pol_t, ..., pol_t-N+1]   (measured)

The regressors `s_t` are measured data. The loss `Σ ||pol_t+1 − var s_t||²` is quadratic in
`var`, so setting its gradient to zero gives a linear equation, the normal equations
`var = (Σ pol_t+1 s_tᵀ)(Σ s_t s_tᵀ)⁻¹`. One `lstsq` call solves it.

**The Kalman predictor.** The controller does, every frame:

    x'_k = A x_k-1                       A = [[var], [I 0]]
    x_k  = x'_k + K (pol_k − var x_k-1)  the update (H x'_k = var x_k-1)
    p_k  = C x_k                         the prediction, delay frames ahead

and the residual it leaves, the thing the score measures, is

    res_t = pol_t − (1 − f) p_t-n − f p_t-n-1        delay = n + f

The regressor of `C` is the state `x_k`, and `x_k` is not measured. The filter computes it from
`var` and `K`. Substitute the first line into the second:

    x_k = (A − K var) x_k-1 + K pol_k = F x_k-1 + K pol_k,     F = A − K var

and unroll it back in time:

    x_k = K pol_k + F K pol_k-1 + F² K pol_k-2 + ...  =  Σ_j F^j K pol_k-j

so

    p_k = Σ_j G_j pol_k-j,        G_j = C F^j K.

**So the predictor is a linear regression of the future `pol` on its past**, like the VAR, but:

- its window is long: every past frame contributes, through powers of `F`, not only the last N;
- its weights `G_j = C (A − K var)^j K` are **products and powers** of the unknowns.

The loss is then a polynomial of high degree in `var`, `K` and `C`, not a quadratic, and its
gradient set to zero is not a linear equation. It has no closed form, so it is minimised
iteratively, by gradient descent.

### Why not fit the `G_j` freely, by least squares?

This is possible: treat each `G_j` as free and solve the linear least squares for them (a FIR
Wiener predictor). That is linear and has a closed form, but at 80 modes with 59 taps it has
59 × 80 × 80 = **377 600 unknowns**. About 3000 training frames hold only 3000 × 80 = 240 000
numbers, so it would overfit.

The Kalman structure ties all the `G_j` to three matrices, 3 × N × n² = **38 400 numbers** for
VAR(2) (25 600 with `learn: "gain"`). That is a strong, physically sensible constraint: the
past acts through an N-frame state. The price of the constraint is the non-linearity.

## 2. Why no closed loop needs simulating

`pol = input / og + applied` is the turbulence whatever the controller did. The controller's
output is added back, so its effect cancels. Recorded once, it can score **any** candidate
predictor offline: run the candidate on `pol` and look at the residual it would have left.

This is also why least squares can be fitted on closed-loop data. The training only needs the
residual a candidate would leave, and that is a function of the recorded `pol` and the
candidate matrices, nothing else.

This holds while `og` is right. When the WFS sees `ρ × og`, part of the output leaks into `pol`
and a real loop appears. That case is handled after training, by an eigenvalue test (section 9),
still without simulating.

## 3. The steps

### Step 1. The inputs

- `pol` `[frames, n]`: the pseudo open loop of the telemetry (mean removed), oldest first.
- A **start** `(var, K, C)`: whichever of the new least-squares + Riccati model and the
  predictor in place does better on the held-out frames ([optimizer, `learn`](rama_controller/varn_lqg/optimizer_varn_lqg.py#L189)).
- `delay = n + f`, which may be fractional.

The start matters because the loss is not convex: gradient descent finds a minimum near where
it starts. The model-based predictor is already a good predictor, so training refines it.

### Step 2. Train / held-out split, and scaling ([L214-L219](rama_controller/varn_lqg/learned_predictor.py#L214-L219))

- The first 70 % of the frames are for training. The last 30 % are held out, never used for the
  gradient, only to decide when to stop and whether the result is better.
- `pol` is divided by its standard deviation. The predictor is linear, so this changes nothing
  in the matrices, but it keeps Adam's step size meaningful whatever the units.

### Step 3. How many past frames count: the taps ([L216-L218](rama_controller/varn_lqg/learned_predictor.py#L216-L218))

The sum `Σ_j G_j pol_k-j` is cut at `taps` terms. `F` is stable, so `F^j` decays like
`radius(F)^j`; `taps` is where that falls below 1e-6, clipped to 50…400 frames (59 in the
notebook). Beyond that the older frames contribute nothing measurable.

### Step 4. One iteration, the gradient step ([L240-L249](rama_controller/varn_lqg/learned_predictor.py#L240-L249))

1. **Pick 512 random target frames** `t` in the training part (a minibatch, only for speed; with
   fewer frames, all of them).
2. **Build the impulse response** from the current matrices ([`_impulse_response`](rama_controller/varn_lqg/learned_predictor.py#L114)):
   `F = A − K var`, then `G_0 = C K`, `G_1 = C F K`, `G_2 = C F² K`, … (`taps` products).
3. **Predict** each target from its past window:
   `applied_t = (1 − f) Σ_j G_j pol_t-n-j + f Σ_j G_j pol_t-n-1-j`, a single `einsum` for the
   whole batch.
4. **Loss** = mean of `(pol_t − applied_t)²` over the batch and the modes. It is the residual the
   controller would leave on those frames.
5. **Backward:** torch differentiates the loss through step 2's chain of products, giving the
   gradient with respect to `var`, `K` and `C` **at the same time**.
6. **Adam step** on all three (on `K` and `C` only with `learn: "gain"`).

No state is carried from one frame to the next here. Each target is predicted directly from
its own window of recorded `pol`, so the frames of a batch are independent, and the step is one
big regression evaluation, like least squares, only non-linear in the unknowns.

### Step 5. Every 25 iterations, the held-out check ([`_residuals`](rama_controller/varn_lqg/learned_predictor.py#L126))

The current matrices are run as the **exact recursive filter**, frame by frame, as the
controller would run them, over the held-out `pol`, from `x = 0`. The first 50 frames are not
counted, while the state forgets its start. The result is the RMS residual on unseen frames.

This is the controller's own recursion over recorded data, still not a closed loop. It uses the
exact filter, not the truncated sum, so the choice does not depend on the truncation.

- If it improved, the matrices are saved as the best so far.
- **Stop** after 20 checks in a row without improvement (500 iterations), or after
  `learn_seconds`.

### Step 6. Safety ([L260-L263](rama_controller/varn_lqg/learned_predictor.py#L260-L263))

The best matrices are returned, not the last ones. If `F` of the result is unstable
(`radius ≥ 1`) or not finite, the start is returned instead.

### Step 7. Outside the training: the robust step and the decision ([optimizer L211-L221](rama_controller/varn_lqg/optimizer_varn_lqg.py#L211-L221))

1. [`robust_mix`](rama_controller/varn_lqg/learned_predictor.py#L78): for an optical-gain error
   ρ in {0.7, 1, 1.3}, the eigenvalues of the loop the controller would close through `pol`
   ([`hidden_loop_radius`](rama_controller/varn_lqg/learned_predictor.py#L51)). If one is ≥ 1,
   the result is moved back towards the start, `(1 − α) start + α learnt`, to the largest stable α.
   This is linear algebra, not a simulation.
2. The result is scored on the held-out frames and **sent only if** it beats the predictor in
   place by `min_gain` (0.5 %).

## 4. Side by side

| | least squares (`fit_var`) | learnt predictor (`train_predictor`) |
|---|---|---|
| unknowns | `var` | `var`, `K`, `C` together |
| fitted on | all frames at once | all training frames, a random 512 per step |
| regressors | the last N frames of `pol`, measured | all past frames, through the filter's state, which depends on the unknowns |
| loss | one-step error of the model | the residual the controller leaves, `delay` frames ahead |
| shape of the loss | quadratic: one exact solution | polynomial: a minimum near the start |
| solved by | normal equations (`lstsq`) | gradient descent (Adam) through `G_j = C F^j K` |
| WFS noise in `pol` | biases `var` (too damped) | does not bias it: white noise can't be predicted |
| `K` and `C` | from `Q`, `R` and the Riccati equation, as right as `Q` and `R` | fitted directly to the score |
| simulated loop | none | none (the robust step uses eigenvalues) |

## 5. Things that could be done differently

- **Full batch instead of minibatches.** All training frames each step, with L-BFGS instead of
  Adam: closer to "everything at once", probably far fewer iterations, but each one costs more.
- **Gradient through the exact recursion** instead of the truncated sum: the same loss without the
  truncation, but a backward pass through thousands of sequential steps, much slower.
- **A one-step loss** (predict `pol_k+1` instead of `pol_k+delay`): this is closer to least
  squares, but it aims at the wrong horizon, and the score is the `delay`-ahead residual.
