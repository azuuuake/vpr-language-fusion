# Nordland Low-LU Failure Taxonomy

This note summarises the manual spot-check of Nordland-clean lowest-LU decile cases where visual top-1 was correct but language-only top-1 was wrong.

Total inspected cases: 11

## Type A: Generic-but-non-place-discriminative captions

Count: 9 / 11

Examples: 1, 2, 3, 5, 7, 8, 9, 10, 11

These cases contain captions that are topically similar, such as railway tracks, snowy mountains, bridges, forests, or trains, but they are not sufficiently place-discriminative. The language retrieves another caption-similar railway frame, but not the correct place match.

Interpretation:
Low LU indicates that the language similarity scores are discriminative, but the discriminative caption signal is not necessarily place-correct.

## Type B: Captioner artifact / degenerate frame captions

Count: 2 / 11

Examples: 4, 6

These cases contain non-scene or anomalous captions such as black background / white text. The language system confidently matches similar artifact captions, but these do not correspond to the true place match.

Interpretation:
This is a captioning-pipeline robustness issue, not a core place-recognition semantic issue.

## Final interpretation

LU measures language-score discriminativeness, not semantic correctness or place correctness.

On AmsterTime and MSLS-val, low LU often corresponds to useful language evidence. On Nordland-clean, low LU can also correspond to generic or artifact-based caption similarity that is confidently separated but place-wrong.
