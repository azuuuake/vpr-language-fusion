# Nordland Representative-Positive Caption Spot-check

This file corrects the earlier manual caption spot-check.

Nordland-clean uses a positive tolerance window rather than one unique positive database image per query.
Therefore, comparing a query caption with only the first positive database frame can be misleading because
the first positive may be an edge-of-window frame.

For each spot-check case, this file records:
- first positive in the tolerance window,
- centre positive in the tolerance window,
- best-visual positive inside the tolerance window,
- language top-1 over the full database,
- language top-1 inside the visual top-10 candidate pool.

The corrected qualitative interpretation should compare the language-selected wrong match against the
representative positive, especially the best-visual positive within the tolerance window.

This affects only the manual qualitative caption analysis. Quantitative R@K retrieval metrics are unaffected
because they are computed against the full positive tolerance window.
