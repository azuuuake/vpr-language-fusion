# Nordland Positive-Window Protocol Note

Nordland-clean was evaluated using a positive-window ground-truth matrix rather than a one-to-one query/database pairing.

## Matrix structure

- Number of queries: 400
- Number of database images: 400
- Mean positives per query: 49.38
- Median positives per query: 51.00
- Minimum positives per query: 26
- Maximum positives per query: 51
- Queries with zero positives: 0
- Queries with multiple positives: 400

## Interpretation

Each query has a tolerance window of positive database frames. Therefore, Nordland-clean should not be interpreted as a dataset with one manually selected positive image per query.

For R@K evaluation, a query is counted as correct if any retrieved database image in the top K belongs to that query's positive window.

Manual inspection of one arbitrary positive frame can be misleading, especially if the selected frame is near the edge of the tolerance window. Therefore, Nordland is reported as a tolerance-window stress-test dataset rather than as a one-to-one manually verified pair dataset.

## Reporting limitation

The Nordland qualitative/caption inspection was conducted as an exploratory diagnostic by the first author. It was not independently annotated, so category counts should not be interpreted as confirmatory qualitative evidence.
