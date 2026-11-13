# Baseline gate: does the simple scorer rank attacks on endpoints it has not seen?

Date: 2026-11-13 · Status: measured and reviewed, task 1.12; owner decisions of 2026-11-13 below · Spec: `.kiro/specs/npu-log-decision/` · Code: `crates/l2d-eval/`

## Answer

**The gate is passed, narrowly and unevenly.** On five folds grouped by URL
path, with the method and path stripped and the default settings, the baseline
scorer reaches a cross-endpoint ROC AUC of 0.919 (mean over the folds; gate
0.90). Three folds are at 0.97 to 0.99. The two folds that hold out
`registro.jsp` or `editar.jsp` are at 0.82, below the bar.

**Owner decisions, 2026-11-13, taken on these figures.**

- The gate figure is the pooled one, as the design says. The gate is passed at
  0.914 in strip mode. With the whole text kept the baseline is below the bar
  (0.897). The mean over the folds is reported beside it.
- The default penalty stays at 1. The comparison of penalties is repeated on
  the AIT data before anything is changed.

The rest of this document is as written before those decisions.

**Which figure is the gate figure was for the owner to decide.** There are two
definitions, and in strip mode the verdict is the same under both.

| Arrangement a, default penalty | strip | keep | isolate |
| --- | ---: | ---: | ---: |
| Mean over the folds of the per-fold figure | 0.9188 (pass) | 0.9182 (pass) | 0.9219 (pass) |
| All held-out predictions pooled into one ranking | 0.9140 (pass) | 0.8966 (below the bar) | 0.9059 (pass) |

- The approved design defines the second: "Metrics per scorer over the pooled
  test predictions" (design.md, Evaluator).
- This document leads with the first because it is the definition the science
  gate document of 2026-11-12 used (pooled per fold, mean over the folds), so
  the baseline and the encoder are stated the same way. That choice is mine
  and is not recorded anywhere before this run.
- `strip` is the mode the gate statement is about ("without memorizing static
  URLs"): it passes under both. With the whole text kept the baseline passes
  under one definition and not under the other.

Three things the owner should know before using either figure.

- **The two weak endpoints are the twin endpoints**, the two that share one
  13-parameter form. With one twin fitted and the other held out, the baseline
  scores 0.82 on the held-out one. With both held out together it scores 0.75
  to 0.76 on each. No penalty repairs that: 0.71 to 0.77 over the whole grid.
- **The penalty is not settled by cross-validation inside the fitting side.**
  The rule the evaluation was written to choose by (smallest inner log loss,
  the rule the encoder's linear head used) picks heavier penalties on the twin
  folds and makes the ranking worse than the default: 0.906 stripped, 0.892
  with the whole text, which fails the gate. A second rule that reads the
  fitting side only (highest inner ROC AUC) picks 0.1 or 0.01 in every fold and
  gives 0.953. Its choice per fold was computed in the first run; its summary
  row was added to the report and to this document after the results had been
  seen. The two rules disagree because they measure different things; see "The
  penalty".
- **Most of what a lighter penalty gains is transfer between the twins.** At
  0.1 the twin folds rise from 0.82 to 0.90. With both twins held out together
  the same change moves the twin fold from 0.762 to 0.774.

The mean over five folds passes in every arrangement at the default penalty,
in all three routing modes. The verdict changes with the way of pooling: over
the five large endpoints, with the twins held out together, the mean is 0.893.

For the science gate: on the science gate's own folds the Rust baseline gives
the figures of the Python re-implementation to the fourth digit (0.9227 against
0.923). On the fold that holds both twins out the baseline scores 0.762 where
the encoder's linear head scored 0.639 and its neighbour ratio 0.504.

Calibration is reported, not a pass condition. It misses the chosen error rate
of 5 % on held-out endpoints for both labels: 14 to 17 % of good logs and 11 to
14 % of bad logs are wrongly ruled out at the default penalty in arrangements
a, b and d, and 83 to 85 % of the good logs of the twin endpoints when both are
unseen.

## The gate figure

Cross-endpoint ROC AUC: the held-out logs of the 8 URL paths that hold both
labels. "Mean" is the mean over the folds of the per-fold figure, the
definition of the science gate document of 2026-11-12; lowest and highest fold
beside it. The design's definition, all held-out predictions as one ranking, is
in the table "the other ways of pooling" below and in the Answer.
All figures count distinct texts. No unit: ROC AUC and average precision are
shares between 0 and 1.

Arrangements (the fold membership is in the report, and below):

- **a**: the task's own. Five folds by URL path of nearly equal numbers of
  logs.
- **b**: the science gate's five folds, reproduced exactly.
- **c**: the science gate's rule with the twins as one unit, so one fold holds
  both out.
- **d**: each of the five large endpoints held out alone, every other path
  fitted.

Penalty rules:

- **default**: L2 penalty 1.0, the scorer's default.
- **inner log loss**: chosen in each fold from 0.01, 0.1, 1, 10, 100 by
  cross-validation on the fitting side, smallest log loss. This is the rule
  the evaluation was written to choose by.
- **inner ROC AUC**: the same cross-validation, highest ROC AUC. The penalty it
  picks in each fold was computed and recorded in the first run. Its summary
  row was added after the results had been seen; it was not the declared rule.

| Arrangement | Routing | Penalty rule | Penalty per fold | Mean | Lowest fold | Highest fold | SD | Average precision, mean | Gate 0.90 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| a | strip | default | 1 | 0.9188 | 0.8191 | 0.9912 | 0.0908 | 0.9446 | pass |
| a | strip | inner log loss | 0.1 / 1 / 10 / 0.1 / 0.1 | 0.9060 | 0.7487 | 0.9924 | 0.1139 | 0.9360 | pass |
| a | strip | inner ROC AUC | 0.01 / 0.1 / 0.1 / 0.01 / 0.01 | 0.9530 | 0.8995 | 0.9925 | 0.0475 | 0.9698 | pass |
| a | keep | default | 1 | 0.9182 | 0.8060 | 0.9979 | 0.1021 | 0.9427 | pass |
| a | keep | inner log loss | 0.01 / 10 / 10 / 0.1 / 0.01 | 0.8915 | 0.7277 | 0.9999 | 0.1453 | 0.9256 | **fail** |
| a | keep | inner ROC AUC | 0.01 / 0.1 / 0.1 / 0.01 / 0.01 | 0.9587 | 0.8994 | 1.0000 | 0.0534 | 0.9713 | pass |
| a | isolate | default | 1 | 0.9219 | 0.8147 | 0.9983 | 0.0978 | 0.9451 | pass |
| b | strip | default | 1 | 0.9227 | 0.8287 | 0.9907 | 0.0855 | 0.9504 | pass |
| b | strip | inner log loss | 0.1 / 1 / 10 / 0.1 / 0.1 | 0.9107 | 0.7622 | 0.9920 | 0.1075 | 0.9428 | pass |
| b | strip | inner ROC AUC | 0.01 / 0.1 / 0.1 / 0.01 / 0.01 | 0.9550 | 0.9045 | 0.9921 | 0.0447 | 0.9725 | pass |
| b | keep | default | 1 | 0.9226 | 0.8166 | 0.9978 | 0.0966 | 0.9491 | pass |
| b | keep | inner log loss | 0.01 / 10 / 10 / 0.01 / 0.01 | 0.8973 | 0.7423 | 1.0000 | 0.1376 | 0.9339 | **fail** |
| b | keep | inner ROC AUC | 0.01 / 0.1 / 0.1 / 0.01 / 0.01 | 0.9609 | 0.9046 | 1.0000 | 0.0505 | 0.9744 | pass |
| b | isolate | default | 1 | 0.9261 | 0.8244 | 0.9983 | 0.0927 | 0.9512 | pass |
| c | strip | default | 1 | 0.9408 | 0.7617 | 0.9907 | 0.1003 | 0.9630 | pass |
| c | strip | inner log loss | 1 / 10 / 10 / 10 / 10 | 0.9285 | 0.7617 | 0.9894 | 0.0955 | 0.9557 | pass |
| c | strip | inner ROC AUC | 0.1 / 0.1 / 0.01 / 0.1 / 0.1 | 0.9456 | 0.7737 | 0.9921 | 0.0962 | 0.9656 | pass |
| c | keep | default | 1 | 0.9436 | 0.7558 | 0.9978 | 0.1052 | 0.9642 | pass |
| c | keep | inner log loss | 1 / 10 / 10 / 100 / 10 | 0.9128 | 0.7558 | 0.9910 | 0.1003 | 0.9455 | pass |
| c | keep | inner ROC AUC | 0.01 / 0.1 / 0.01 / 0.1 / 0.1 | 0.9489 | 0.7651 | 1.0000 | 0.1029 | 0.9608 | pass |
| c | isolate | default | 1 | 0.9448 | 0.7591 | 0.9983 | 0.1041 | 0.9649 | pass |
| d | strip | default | 1 | 0.9186 | 0.8191 | 0.9907 | 0.0906 | 0.9443 | pass |
| d | strip | inner log loss | 10 / 0.1 / 0.1 / 0.1 / 1 | 0.9060 | 0.7487 | 0.9920 | 0.1139 | 0.9358 | pass |
| d | strip | inner ROC AUC | 0.1 / 0.01 / 0.01 / 0.01 / 0.1 | 0.9530 | 0.8995 | 0.9921 | 0.0475 | 0.9696 | pass |
| d | keep | default | 1 | 0.9183 | 0.8060 | 0.9978 | 0.1021 | 0.9427 | pass |
| d | keep | inner log loss | 10 / 0.01 / 0.01 / 0.01 / 10 | 0.8917 | 0.7273 | 1.0000 | 0.1457 | 0.9256 | **fail** |
| d | keep | inner ROC AUC | 0.1 / 0.01 / 0.01 / 0.01 / 0.1 | 0.9589 | 0.8993 | 1.0000 | 0.0537 | 0.9714 | pass |
| d | isolate | default | 1 | 0.9221 | 0.8146 | 0.9983 | 0.0981 | 0.9453 | pass |

"Without memorizing static URLs": `strip` drops the method and the path before
anything is hashed, so the scorer cannot use the URL at all; that is the mode
the gate statement is about. `keep` and `isolate` could use the path, but a
held-out path was never seen, and they score the same as `strip` to within
0.004 on the mean.

In arrangement c the mean looks better than in a and b. That is the make-up of
the folds, not a better scorer: the two weak endpoints share one fold, so four
of the five folds are easy, and the fifth of them holds one small endpoint with
4 good logs.

### Per fold

Stripped, the cross-endpoint ROC AUC of each fold with the endpoints it holds
out.

| Arrangement | Fold | Held-out endpoints with both labels | Held-out logs | Default | Inner log loss | Inner ROC AUC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| a | 0 | `autenticar` | 5,122 | 0.9749 | 0.9795 | 0.9806 |
| a | 1 | `registro` | 5,122 | 0.8201 | 0.8201 | 0.9028 |
| a | 2 | `editar` | 5,122 | 0.8191 | 0.7487 | 0.8995 |
| a | 3 | `pagar`, `entrar` | 5,121 | 0.9912 | 0.9924 | 0.9925 |
| a | 4 | `anadir`, `caracteristicas`, `vaciar` | 5,121 | 0.9887 | 0.9892 | 0.9894 |
| b | 0 | `autenticar`, `caracteristicas` | 5,371 | 0.9764 | 0.9797 | 0.9802 |
| b | 1 | `registro`, `entrar` | 5,271 | 0.8298 | 0.8298 | 0.9081 |
| b | 2 | `editar`, `vaciar` | 5,264 | 0.8287 | 0.7622 | 0.9045 |
| b | 3 | `pagar` | 4,925 | 0.9907 | 0.9920 | 0.9921 |
| b | 4 | `anadir` | 4,777 | 0.9880 | 0.9898 | 0.9899 |
| c | 0 | `registro`, `editar`, `entrar` | 10,058 | 0.7617 | 0.7617 | 0.7737 |
| c | 1 | `autenticar`, `vaciar` | 5,273 | 0.9759 | 0.9643 | 0.9804 |
| c | 2 | `pagar` | 4,925 | 0.9907 | 0.9894 | 0.9921 |
| c | 3 | `anadir` | 4,778 | 0.9880 | 0.9394 | 0.9898 |
| c | 4 | `caracteristicas` | 574 | 0.9878 | 0.9878 | 0.9919 |
| d | 0 | `editar` | 4,774 | 0.8191 | 0.7487 | 0.8995 |
| d | 1 | `anadir` | 4,453 | 0.9880 | 0.9898 | 0.9899 |
| d | 2 | `autenticar` | 4,783 | 0.9749 | 0.9795 | 0.9806 |
| d | 3 | `pagar` | 4,600 | 0.9907 | 0.9920 | 0.9921 |
| d | 4 | `registro` | 4,781 | 0.8201 | 0.8201 | 0.9028 |

### Per endpoint, and the other ways of pooling

Stripped, default penalty. Each endpoint is scored by the fold that held it
out. The three small endpoints have 2 to 4 good logs, so their ROC AUC rests
on 2 to 4 texts.

| Endpoint | Good logs | Bad logs | ROC AUC, a | ROC AUC, c (twins out together) | Average precision, a |
| --- | ---: | ---: | ---: | ---: | ---: |
| `/tienda1/publico/pagar.jsp` | 1,940 | 2,660 | 0.9907 | 0.9907 | 0.9954 |
| `/tienda1/publico/anadir.jsp` | 1,676 | 2,777 | 0.9877 | 0.9880 | 0.9945 |
| `/tienda1/publico/autenticar.jsp` | 2,000 | 2,783 | 0.9749 | 0.9749 | 0.9847 |
| `/tienda1/publico/registro.jsp` | 2,000 | 2,781 | 0.8201 | 0.7533 | 0.8737 |
| `/tienda1/miembros/editar.jsp` | 2,000 | 2,774 | 0.8191 | 0.7569 | 0.8732 |
| `/tienda1/publico/caracteristicas.jsp` | 4 | 246 | 0.9878 | 0.9878 | 0.9998 |
| `/tienda1/publico/entrar.jsp` | 2 | 163 | 0.9969 | 0.9969 | 0.9999 |
| `/tienda1/publico/vaciar.jsp` | 2 | 163 | 0.9969 | 0.9969 | 0.9999 |

| Way of pooling, stripped, default penalty | a | b | c | d |
| --- | ---: | ---: | ---: | ---: |
| Mean over the folds (the gate figure) | 0.9188 | 0.9227 | 0.9408 | 0.9186 |
| Mean over the five large endpoints | 0.9185 | 0.9185 | 0.8928 | 0.9186 |
| Mean over all eight endpoints | 0.9468 | 0.9468 | 0.9307 | n/a |
| All folds as one ranking | 0.9140 | 0.9181 | 0.8678 | 0.9149 |
| Overall, single-label paths included, mean over the folds | 0.9262 | 0.9290 | 0.9245 | 0.9186 |

"All folds as one ranking" puts ratios from five different fits into one list,
so a shift of the ratios between folds counts against it. It is below the bar
in arrangement c (0.8678) and, with the whole text kept, also in a (0.8966),
b (0.8993) and d (0.8937).

## Against the encoder, on the same held-out logs

Arrangement b is the science gate's fold design. Its five lists of held-out
paths have the SHA-256 that the science gate's run records carry, and the same
numbers of held-out logs (5,371, 5,271, 5,264, 4,925, 4,777), so the figures
below are on identical logs. The encoder columns are copied from
`docs/2026-11-12-science-gate-encoder.md` (8-bit model, input stripped, mean of
three seeds); nothing of the encoder was run here.

| Fold | Held-out endpoints | Baseline, default (this run) | Baseline, Python re-implementation | Baseline, inner log loss | Baseline, inner ROC AUC | Encoder, linear head | Encoder, neighbour ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `autenticar`, `caracteristicas` | 0.9764 | 0.976 | 0.9797 | 0.9802 | 0.974 | 0.904 |
| 1 | `registro`, `entrar` | 0.8298 | 0.830 | 0.8298 | 0.9081 | 0.935 | 0.907 |
| 2 | `editar`, `vaciar` | 0.8287 | 0.829 | 0.7622 | 0.9045 | 0.949 | 0.919 |
| 3 | `pagar` | 0.9907 | 0.991 | 0.9920 | 0.9921 | 0.991 | 0.991 |
| 4 | `anadir` | 0.9880 | 0.988 | 0.9898 | 0.9899 | 0.945 | 0.745 |
| Mean | | 0.9227 | 0.923 | 0.9107 | 0.9550 | 0.959 | 0.893 |

The Rust tool and the Python re-implementation agree on every fold to the
digits the science gate printed; against its run records the largest
difference on a fold is about 0.00001, stripped and kept (two solvers with different
stopping rules). Its statement that the baseline averages 0.955
at penalty 0.1 is confirmed as a sensitivity figure: 0.9548 here.

With both twins held out together (arrangement c, fold 0: `registro`, `editar`
and the small `entrar`, 10,058 held-out logs):

| Scorer | Cross-endpoint ROC AUC |
| --- | ---: |
| Baseline, default penalty, stripped (this run) | 0.7617 |
| Baseline, whole text kept (this run) | 0.7558 |
| Baseline over the penalty grid 0.01 / 0.1 / 1 / 10 / 100, stripped (sensitivity) | 0.7697 / 0.7737 / 0.7617 / 0.7368 / 0.7080 |
| Baseline, as the review of the science gate reported it | 0.762 |
| Encoder, linear head, one seed (review of the science gate) | 0.639 |
| Encoder, neighbour ratio, one seed (review of the science gate) | 0.504 |

The review's run with both twins held out left no run record, so its exact
list of held-out logs cannot be compared. Its baseline figure matches this
fold to the third digit.

What the science-gate decision can take from this:

- The baseline side of the comparison stands as the science gate stated it.
  The Rust tool gives the same figures.
- At the default penalty the linear head leads by 0.036 on the five folds
  (0.959 against 0.923). All of that lead is on the two twin folds.
- "The baseline at a lighter penalty is level with the head" holds if the
  penalty is chosen by inner ROC AUC (0.955 against 0.959). It does not hold
  under the inner log-loss rule, which is the rule the head itself was tuned
  with (0.911).
- On endpoints whose form is unseen the baseline is weak (0.76) and the encoder
  was weaker (0.64, one seed). Neither clears 0.90 there.

## The penalty

The inner cross-validation cuts the fitting side into four folds by URL path
and fits every penalty on three of them. Stripped, arrangement b; the other
arrangements show the same pattern and are in the report.

| Fold | Held out | Inner log loss at 0.01 / 0.1 / 1 / 10 / 100 | Chosen by log loss | Inner ROC AUC at 0.01 / 0.1 / 1 / 10 / 100 | Chosen by ROC AUC |
| ---: | --- | --- | ---: | --- | ---: |
| 0 | `autenticar` | 0.379 / **0.328** / 0.419 / 0.512 / 0.590 | 0.1 | **0.9478** / 0.9424 / 0.9011 / 0.8502 / 0.8099 | 0.01 |
| 1 | `registro` | 0.995 / 0.659 / **0.472** / 0.476 / 0.568 | 1 | 0.9309 / **0.9323** / 0.9269 / 0.9018 / 0.8662 | 0.1 |
| 2 | `editar` | 0.999 / 0.690 / 0.497 / **0.481** / 0.569 | 10 | 0.9304 / **0.9309** / 0.9262 / 0.9034 / 0.8642 | 0.1 |
| 3 | `pagar` | 0.544 / **0.385** / 0.430 / 0.511 / 0.583 | 0.1 | **0.9485** / 0.9446 / 0.9052 / 0.8440 / 0.7947 | 0.01 |
| 4 | `anadir` | 0.360 / **0.304** / 0.382 / 0.493 / 0.581 | 0.1 | **0.9472** / 0.9439 / 0.9042 / 0.8638 / 0.8204 | 0.01 |

Log loss is in nats per log; lower is better.

Why the two rules disagree. Log loss asks whether the ratio is a good
probability; ROC AUC asks only whether bad logs rank above good ones. On an
endpoint the scorer has not seen, the ratios shift as a whole (task 1.7 saw
good logs average 0.65 there). A light penalty makes the scorer more certain,
so a shifted ratio costs more log loss even where the ranking is better. The
likely reason for folds 1 and 2, not tested on its own: their fitting side
holds one twin, and one inner fold holds that twin out with no twin left to
learn from, which is where a heavy penalty wins on log loss.

Sensitivity on the held-out side. This is every penalty of the grid scored
against held-out labels. It is not a setting: choosing from it is choosing with
the held-out figures in view.

| Penalty | a, strip | a, keep | b, strip | b, keep | c, strip | c, keep | d, strip | d, keep |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.9560 | 0.9697 | 0.9579 | 0.9714 | 0.9450 | 0.9505 | 0.9561 | 0.9700 |
| 0.1 | 0.9527 | 0.9582 | 0.9548 | 0.9605 | 0.9455 | 0.9505 | 0.9527 | 0.9583 |
| 1 (default) | 0.9188 | 0.9182 | 0.9227 | 0.9226 | 0.9408 | 0.9436 | 0.9186 | 0.9183 |
| 10 | 0.8782 | 0.8707 | 0.8828 | 0.8753 | 0.9235 | 0.9190 | 0.8769 | 0.8690 |
| 100 | 0.8356 | 0.8204 | 0.8398 | 0.8243 | 0.8896 | 0.8744 | 0.8317 | 0.8149 |

What the default should be, on this evidence. This task does not change it.

- For ranking inside a fold, a lighter penalty is better or equal: the inner
  ROC AUC prefers 0.1 or 0.01 to 1.0 in every fold of every arrangement and
  both routing modes, without looking at held-out labels, and the held-out
  sensitivity per fold agrees. 0.1 is the cautious choice of the two: it is
  never more than 0.012 behind 0.01 on a mean, and its inner log loss is far
  lower.
- That holds per fold only. With all folds pooled into one ranking (the
  design's definition) the inner-ROC-AUC rule, stripped, is below the default:
  0.9093 against 0.9140 in a, 0.9122 against 0.9181 in b, 0.9071 against 0.9149
  in d. A penalty that differs from fold to fold shifts the ratios between
  folds.
- The size of the gain depends on the twins. With one twin fitted: +0.03 on
  the mean, +0.08 on the twin folds. With both unseen: +0.005 on the mean,
  +0.012 on the twin fold.
- For ratios read as probabilities, 1.0 or heavier is better on unseen forms:
  that is what the log-loss rule says. Triage ranks by ratio and calibrates
  afterwards, so ranking is the property the tool uses.
- So the evidence supports 0.1 as the default if the owner accepts inner ROC
  AUC as the criterion. It does not support 0.1 under the log-loss criterion the
  evaluation was written with. That is the owner's decision; picking the criterion
  after seeing which one gives the better held-out figure is itself a choice
  made with the held-out figures in view.
- A lighter penalty costs iterations, not convergence. All 1,020 fits of the
  report ended as `converged` with the gradient criterion met; none reached the
  cap of 1,000. Mean iterations per fit: 643 at 0.01, 347 at 0.1, 147 at 1, 57
  at 10, 24 at 100. The most was 856 (strip, 0.01).

## Calibration

In each fold the first inner fold of the fitting side is set aside as
calibration logs: whole URL paths the calibrated scorer was not fitted on. That
scorer is fitted on the rest of the fitting side, calibrated at error rate
0.05, and judges the fold's held-out logs. The achieved error rate of a label
is the share of its logs whose label set does not contain it. Requirement 11.8
asks for these figures; they are not a pass condition.

All folds together, stripped:

| Arrangement | Penalty | Good logs wrongly ruled out | Bad logs wrongly ruled out | Set good | Set bad | Set both | Set neither |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a | default | 16.7 % | 11.7 % | 32.9 % | 56.8 % | 10.3 % | 0.0 % |
| a | inner log loss | 6.7 % | 12.3 % | 38.1 % | 52.8 % | 9.1 % | 0.0 % |
| b | default | 15.2 % | 11.3 % | 33.9 % | 56.5 % | 9.6 % | 0.0 % |
| b | inner log loss | 6.7 % | 12.1 % | 38.4 % | 53.0 % | 8.6 % | 0.0 % |
| c | default | 34.9 % | 4.0 % | 22.7 % | 52.1 % | 25.3 % | 0.0 % |
| c | inner log loss | 35.0 % | 2.1 % | 18.3 % | 57.4 % | 24.3 % | 0.0 % |
| d | default | 14.3 % | 13.4 % | 37.7 % | 52.0 % | 10.2 % | 0.0 % |
| d | inner log loss | 6.3 % | 14.5 % | 42.7 % | 48.5 % | 8.7 % | 0.0 % |

Shares are of 25,608 held-out logs in a, b and c (9,644 good, 15,964 bad) and
of 23,391 in d (9,616 good, 13,775 bad). With the whole text kept the picture
is the same (a, default: 15.7 % and 12.4 %; c, default: 39.2 % and 6.5 %).

Per group, arrangement a, stripped, default penalty:

| Group | Good logs | Good wrongly ruled out | Bad logs | Bad wrongly ruled out |
| --- | ---: | ---: | ---: | ---: |
| `/tienda1/publico/pagar.jsp` | 1,940 | 0.0 % | 2,660 | 2.0 % |
| `/tienda1/publico/autenticar.jsp` | 2,000 | 1.6 % | 2,783 | 2.3 % |
| `/tienda1/publico/anadir.jsp` | 1,676 | 64.6 % | 2,777 | 0.2 % |
| `/tienda1/publico/registro.jsp` | 2,000 | 10.4 % | 2,781 | 34.2 % |
| `/tienda1/miembros/editar.jsp` | 2,000 | 13.1 % | 2,774 | 28.4 % |
| `caracteristicas`, `entrar`, `vaciar` | 8 | 100.0 % | 572 | 0.0 % |
| Single-label paths, 1,624 of them | 20 | 100.0 % | 1,617 | 0.0 % |

With both twins held out (arrangement c): 82.9 % of the good logs of `editar`
and 84.8 % of `registro` are wrongly ruled out, and 0.6 % and 0.4 % of their
bad logs.

The stated error rate holds for logs that resemble the calibration logs
(requirement 11.9). Held-out endpoints do not, and the miss goes both ways
here: on some endpoints good logs are ruled out, on the twins bad logs are
called good. Task 1.13 saw only the first kind. Stripped, no log was given the
empty set; with the whole text kept and the log-loss penalty, 0.4 % to 1.4 % of
the logs were.

## Single-label paths

1,624 of the 1,632 URL paths hold one label: 1,604 hold only bad logs and 20
only good logs, 1,637 logs in all (20 good, 1,617 bad). They are in no
cross-endpoint figure. Each fold's held-out side is ranked by ratio as a whole,
and the table gives the share of the single-label logs inside two review
budgets. Default penalty.

| Arrangement | Routing | Budget: 10 % of the held-out logs | Budget: as many logs as there are bad logs | Recall of all bad logs at the 10 % budget | Recall at the bad-count budget |
| --- | --- | ---: | ---: | ---: | ---: |
| a | strip | 19.8 % | 100.0 % | 0.1602 | 0.9024 |
| a | keep | 15.8 % | 100.0 % | 0.1605 | 0.9008 |
| a | isolate | 13.3 % | 100.0 % | 0.1605 | 0.9056 |
| b | strip | 3.7 % | 100.0 % | 0.1601 | 0.9056 |
| b | keep | 10.2 % | 100.0 % | 0.1602 | 0.9046 |
| b | isolate | 2.0 % | 100.0 % | 0.1602 | 0.9095 |
| c | strip | 0.0 % | 99.9 % | 0.1493 | 0.9310 |
| c | keep | 0.9 % | 98.2 % | 0.1493 | 0.9334 |
| c | isolate | 0.0 % | 99.9 % | 0.1493 | 0.9357 |

Arrangement d holds out no single-label path. About 62 % of the held-out logs
are bad, so a budget of 10 % can reach a recall of 0.16 at most; the scorer
reaches it, which says the top tenth of each ranking is almost all bad. The
single-label logs are rarely in that top tenth and nearly always within the
larger budget: the scorer ranks them as bad, but not as the most certain. A tie
at the budget is shared between the tied logs.

## Cost and memory

Reference host: AMD Ryzen AI 9 HX 370, 12 cores, 29.6 GiB of memory. Release
build, one process.

Measured on one thread before the folds start, on fold 0 of arrangement a
(20,486 fitting logs, 5,122 held-out logs), default penalty:

Three runs have a record on disk. The host was loaded by other work during all
of them (another agent was compiling), so the cost is a range, not one figure.

| Run record | Workers | Fit, wall time: strip / keep / isolate | Scoring: strip / keep / isolate | All folds |
| --- | ---: | --- | --- | ---: |
| `reports\2026-11-13-csic-2010-baseline.run.json` | 6 | 2.87 / 4.74 / 5.98 s | 210,570 / 112,017 / 109,756 logs/s | 1,820 s |
| `reports\rerun-check\2026-11-13-csic-2010-baseline.run.json` | 6 | 4.31 / 7.21 / 7.66 s | 147,380 / 84,708 / 88,858 logs/s | 1,697 s |
| `D:\BuildArtifacts\log2decision\review-1.12\rerun\` (the reviewer's run) | 4 | 4.10 / 5.80 / 8.82 s | 167,170 / 63,442 / 66,805 logs/s | 1,769 s |

- Fit of 20,486 logs, one thread: 2.9 to 4.3 s stripped, 4.7 to 7.2 s kept,
  6.0 to 8.8 s isolate. Iterations are the same in every run: 163, 201, 155.
- Scoring, one thread: 147,000 to 211,000 logs/s stripped, 63,000 to 112,000
  kept, 67,000 to 110,000 isolate. It is the scores frame of the held-out side,
  the median of three passes.
- The first ratio after a fit took 0.02 to 0.05 ms in the first record.
- The whole evaluation is 1,020 fits: 1,697 to 1,820 s on 6 worker threads,
  1,769 s on 4.
- The runs were not on a quiet machine. The spread between records, up to a
  factor of 1.8 on one figure, is the load of the host and says nothing about
  the scorer. No quiet-machine figure exists.
- Process CPU time, the energy proxy of requirement 9.4, was not measured: the
  workspace does not link the Windows call for it. The cost section runs on
  one thread, so its CPU time is at most its wall time.

Memory (requirement 12). In the first record the limit was 5,854 MiB, half of
the 11.4 GiB that were free at the start. Every fit and every calibration asks
the budget before it allocates; each worker thread has an equal share of the
limit (967 MiB) and its fits are checked against that share. The fits of the
cost section asked for 112 MiB (strip), 144 MiB (keep) and 175 MiB (isolate).
The heap of the process, counted by a counting allocator, peaked at 1,072 MiB
with 6 workers, 18 % of the limit (1,069 MiB in the second record; 766 MiB with
4 workers in the reviewer's run). The peak grows with the number of workers.

## How it was measured

```
cargo run --release -j 4 -p l2d-eval --example baseline_gate -- --workers 6
```

from the repository root. It reads `data/csic-2010/csic_database.csv` (SHA-256
`c420f0bc0464376de75b6c419a0ac226fe69fe12c8ac4908843273721e44e637`, 61,065
records, 25,608 distinct texts: 9,644 good, 15,964 bad) and writes three files
into `reports/`, which is not tracked:

- `2026-11-13-csic-2010-baseline.json`: every figure, per fold, per endpoint and
  over the folds, the settings, the dataset and its hash, and the full list of
  held-out paths of every fold.
- `2026-11-13-csic-2010-baseline.md`: the same as tables.
- `2026-11-13-csic-2010-baseline.run.json`: times, cost and memory of the run.

The first two hold nothing that depends on the clock beyond the date, on the
number of workers or on the host's memory. A second run wrote the same bytes:
SHA-256 `a6f7ec601d754cbd671d2b610e0e049eb7c4447dde31f29d8a0d7bb772d69f8e` for the JSON file in both.

Settings: 65,536 buckets, byte 1- to 3-grams, tolerance 0.001, at most 1,000
iterations, all the scorer's defaults. Penalty grid 0.01, 0.1, 1, 10, 100.
Four inner folds. Error rate 0.05, a group list from 100 calibration logs of a
label. Review budget 10 % of the held-out logs. `isolate` was run at the
default penalty only, without the choice of the penalty and without
calibration.

### The folds

All folds are by URL path, fixed by rule from the paths and their numbers of
logs before anything is fitted. Arrangement a reads no label. Arrangements b
and c use which paths hold both labels to deal the large endpoints one per
fold, as the science gate did; d uses it to pick the endpoints to hold out.

| Arrangement | Fold | Held-out paths | Held-out logs | Endpoints with both labels |
| --- | ---: | ---: | ---: | --- |
| a | 0 | 340 | 5,122 | `autenticar` |
| a | 1 | 342 | 5,122 | `registro` |
| a | 2 | 349 | 5,122 | `editar` |
| a | 3 | 345 | 5,121 | `pagar`, `entrar` |
| a | 4 | 256 | 5,121 | `anadir`, `caracteristicas`, `vaciar` |
| b | 0 | 327 | 5,371 | `autenticar`, `caracteristicas` |
| b | 1 | 327 | 5,271 | `registro`, `entrar` |
| b | 2 | 327 | 5,264 | `editar`, `vaciar` |
| b | 3 | 326 | 4,925 | `pagar` |
| b | 4 | 325 | 4,777 | `anadir` |
| c | 0 | 328 | 10,058 | `registro`, `editar`, `entrar` |
| c | 1 | 327 | 5,273 | `autenticar`, `vaciar` |
| c | 2 | 326 | 4,925 | `pagar` |
| c | 3 | 326 | 4,778 | `anadir` |
| c | 4 | 325 | 574 | `caracteristicas` |
| d | 0 to 4 | 1 each | 4,774 / 4,453 / 4,783 / 4,600 / 4,781 | `editar` / `anadir` / `autenticar` / `pagar` / `registro` |

- a: paths from the largest to the smallest, each to the fold with the fewest
  logs so far (the rule of scikit-learn's `GroupKFold`).
- b: the paths with both labels dealt in turn from the largest to the
  smallest, then the single-label paths dealt in turn in byte order
  (`cv5_fold_of_group` of `tools/science/l2dsci.py`).
- c: rule b with `registro.jsp` and `editar.jsp` as one unit. They also stay
  together in the inner folds.
- d: a path is held out alone if it has at least 100 good and 100 bad logs.
  Too small: `caracteristicas.jsp` (4 good), `entrar.jsp` (2), `vaciar.jsp`
  (2). In d a twin is held out while the other is fitted.

### How leakage is kept out

- The logs are distinct texts. The reader deduplicates before anything is
  split; no text is on two sides.
- A split is by URL path only. The fold evaluation refuses a mask that puts
  one path on both sides.
- The choice of the penalty and the calibration are functions that are handed
  a copy of the fitting-side rows and nothing else.
- Before a scorer scores a log, the evaluation asks the scorer whether it was
  fitted on that text (the per-log check of task 1.11). A yes stops the run.
- Nothing is fitted, scaled or chosen on all logs. Feature rows are hashed and
  normalized per text; there is no vocabulary.
- Held-out labels are read by the metrics and by the count of calibrated
  decisions, by nothing else. Tests flip every held-out label and change every
  held-out text and check that the choice of the penalty, every fit and every
  ratio stay the same bit for bit.
- Which paths hold both labels is read from all labels. It decides which logs
  a metric is computed on, and in b, c and d how folds are dealt. It reaches no
  fit.

## Limits

- One dataset, eight endpoints with both labels, five of them large enough to
  rank. A mean over five folds with a standard deviation of 0.09 is a pattern,
  not a significance test. No confidence interval was computed.
- CSIC reuses a small attack dictionary across endpoints (task 1.7: 28 % of
  held-out bad logs carry a parameter value seen only in bad fitting logs).
  "Unseen endpoint" here means unseen path and form, not unseen attacks.
- The two penalty rules disagree, and which one to trust was not fixed in
  advance beyond "log loss". The inner ROC AUC figures are honest
  cross-validation, but their summary row was added, and the preference for
  them arose, after the results were seen.
- Which definition of the gate figure applies (mean over the folds, or the
  design's pooled ranking) is not recorded anywhere before this run.
- The held-out check compares whole texts. In strip mode two requests with the
  same payload under different paths are different texts but the same input to
  the scorer; such pairs across the two sides of a split are not detected or
  counted, and how many there are was not measured.
- The inner folds and the calibration part follow from one rule
  (`GroupKFold` on the fitting side, first inner fold for calibration). Other
  cuts were not tried.
- The calibrated decisions come from a scorer fitted on about three quarters of
  the fitting side, not from the scorer whose ROC AUC is reported.
- The comparison with the review's twin run rests on matching figures, not on
  a shared list of held-out logs.
- CPU time was not measured, and every cost figure is from a loaded machine.
- Not run: the encoder, the neighbour scorer, the `l2d eval` command (task
  3.9). The pieces in `crates/l2d-eval/` are what task 3.7 generalizes.
