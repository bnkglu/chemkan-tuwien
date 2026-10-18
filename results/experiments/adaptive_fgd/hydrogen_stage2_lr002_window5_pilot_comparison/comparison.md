# Hydrogen Stage 2 FGD pilot comparison

These runs start from the same author-matched Stage 1 archive and train in normalized coordinates. The complete RHS is divided by the saved t_ref exactly once.

The pilots use initial learning rate 0.002, Tsit5 rtol=1e-4/atol=1e-6, a five-loss nonmonotone line-search window, and Armijo c1=1e-6. These are learning-rate and numerical-solver settings, not starting-loss targets.

All reported losses and MSE contributions below are divided by the number of observation time points (100 here). The conservation penalty is divided by the same count to preserve its relative weight. Physical temperature metrics are unchanged. This report conversion does not retrain the models.

The functional-gradient sample centers are now clipped to [-1,1], matching the additive edge functions' evaluation coordinates. Comparisons against older runs are confounded by this correction as well as the changed optimization settings.

Fixed-grid FGD is an approximation on a fixed interpolation grid and uses the same line search. It does not enforce adaptive representation certification. Both methods still reject nonfinite trajectories and stop if no acceptable representable update is found.

Shared initial model hash: **True**. Shared dataset hash: **True**. Shared Stage 1 archive hash: **True**.

| Run | Completed / requested | Certified updates | Temporary train-loss increases | Recorded solver trial failures | Stop |
|---|---:|---:|---:|---:|---|
| hydrogen_stage2_current100_lr002_window5_adaptive_100steps_bounded | 100 / 100 | 100 | 22 | 5 | step_limit |
| hydrogen_stage2_current100_lr002_window5_fixed256_100steps | 100 / 100 | 100 | 31 | 0 | step_limit |

Evaluation solves have a finite work cap: max(50,000, training cap) RHS evaluations per solve. Hitting that cap is recorded as an evaluation failure, rather than producing an unbounded solve or a fabricated loss. Trial-failure counts include only diagnostics attached to saved updates; an interrupted or rejected final update can have no saved trial diagnostics.

- hydrogen_stage2_current100_lr002_window5_adaptive_100steps_bounded: original training reduction `sum_over_time`; evaluation cap 50,000; training cap 10,000 (0 means unbounded).
  Resumed from /Users/berke/projects/chemkan-with-functional-gradient-update/results/experiments/adaptive_fgd/hydrogen_stage2_current100_lr002_window5_adaptive_100steps/checkpoint_progress.pt at update 1; elapsed offset 21.5549 s. Recorded timing excludes work performed after the source checkpoint in its original process.
- hydrogen_stage2_current100_lr002_window5_fixed256_100steps: original training reduction `sum_over_time`; evaluation cap 50,000; training cap 10,000 (0 means unbounded).

Missing post-update loss/evaluation records are left as gaps in the history. The saved weights are still evaluated directly below, so incomplete logging does not cause an older checkpoint to be substituted.

## Training solver

| Run | Split | Total loss | Species MSE contribution | Temperature MSE contribution | Temperature MAE (K) | Predicted / reference median peak rise (K) |
|---|---|---:|---:|---:|---:|---:|
| hydrogen_stage2_current100_lr002_window5_adaptive_100steps_bounded | train | 0.060716 | 0.0495445 | 0.0111684 | 479.462 | 1088.17 / 1551.82 |
| hydrogen_stage2_current100_lr002_window5_adaptive_100steps_bounded | test | 0.0373946 | 0.0298408 | 0.0075504 | 461.699 | 1435.46 / 1611.58 |
| hydrogen_stage2_current100_lr002_window5_fixed256_100steps | train | 0.049148 | 0.0402419 | 0.00890399 | 412.887 | 1394.64 / 1551.82 |
| hydrogen_stage2_current100_lr002_window5_fixed256_100steps | test | 0.0179059 | 0.0115393 | 0.00636465 | 410.207 | 1466.03 / 1611.58 |

## Strict solver reevaluation

| Run | Split | Total loss | Species MSE contribution | Temperature MSE contribution | Temperature MAE (K) | Predicted / reference median peak rise (K) |
|---|---|---:|---:|---:|---:|---:|
| hydrogen_stage2_current100_lr002_window5_adaptive_100steps_bounded | train | 0.0607203 | 0.0495435 | 0.0111737 | 479.64 | 1088.57 / 1551.82 |
| hydrogen_stage2_current100_lr002_window5_adaptive_100steps_bounded | test | 0.0373975 | 0.0298393 | 0.00755471 | 461.878 | 1435.16 / 1611.58 |
| hydrogen_stage2_current100_lr002_window5_fixed256_100steps | train | 0.0491459 | 0.040236 | 0.00890777 | 413.159 | 1394.03 / 1551.82 |
| hydrogen_stage2_current100_lr002_window5_fixed256_100steps | test | 0.0179088 | 0.0115438 | 0.00636306 | 410.15 | 1466.15 / 1611.58 |

Species and temperature MSE contributions both retain division by all ten state outputs and averaging over observation time points; their sum is the trajectory MSE. Total loss adds the weighted time-averaged elemental conservation penalty. Peak rise is max(T(t)) − T(0), computed per condition and then summarized by its median. Test curves and losses describe only the saved final checkpoint, with no selection using the held-out condition.

The loss plot uses a logarithmic loss axis and logged training-solver objectives converted to the time-averaged reduction after each update. Its strict-evaluation markers reevaluate only the final saved model. The temperature plot shows the median test trajectory (one held-out condition in the current dataset), with both solver tolerances.

The pilots overlapped on the same machine, so their recorded elapsed times are not an isolated speed comparison. A short pilot cannot establish convergence or show that longer temperature training will succeed. The Stage 1 archive and the single held-out condition remain limitations of these comparisons.

[Loss vs epochs](loss_vs_epochs.png) · [Test temperature](test_temperature.png) · [Machine-readable comparison](comparison.json)
