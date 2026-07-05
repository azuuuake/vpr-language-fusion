# V2 Calibration Selection Rule Lock

Created before corrected recalibration/evaluation.

Timestamp UTC: 2026-07-05T07:07:06.230674+00:00

## Fixed calibration split

Dataset: MSLS-val  
Calibration fraction: 0.30  
Seed: 42  
Existing split file: results/redesign_v2/msls_calibration_split_indices.npz  

The calibration split must not be regenerated or changed.

## Locked selection rule

Hyperparameters are selected using the MSLS-val calibration split only.

Selection order:

1. Maximise calibration R@1.
2. If R@1 is tied, choose the smaller candidate_pool.
3. If still tied, choose the smaller rho.

R@5 and R@10 are reported but are not used for tie-breaking.

## Search space

rho ∈ [1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 20, 30]  
candidate_pool ∈ [10, 20, 50, 100]  
uncertainty_k = 10

After this selection, rho and candidate_pool are frozen for final evaluation.
