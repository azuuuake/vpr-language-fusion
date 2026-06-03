# VPR Language Fusion

This repository contains reproducibility files for an early checkpoint in a visual-confidence-aware language fusion project for Visual Place Recognition.

## Purpose

The goal of this checkpoint is to verify whether the descriptor-derived similarity matrix from `auto_VPR` is already sorted before computing uncertainty metrics such as RS, SD, and SU.

## Main finding

The descriptor-derived similarity matrix is **not sorted** by retrieval score. Its columns follow database image order.

Therefore, before computing uncertainty metrics, we must use:

```python
sim_matrix_sorted = sort_scores_desc(sim_matrix)
