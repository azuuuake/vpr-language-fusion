# VPR Language Fusion

This repository contains reproducibility files for an early checkpoint in a **visual-confidence-aware language fusion project** for Visual Place Recognition (VPR).

The project investigates whether visual retrieval uncertainty can be used to decide when language-based scene descriptions should help re-rank visual place recognition results.

---

## Project Goal

Visual Place Recognition systems retrieve the most likely database image for a given query image. However, visual-only retrieval can fail when places look similar, when scenes change over time, or when the top visual candidates are very close in similarity score.

This project explores a confidence-gated fusion idea:

> Use visual descriptors first.  
> Measure visual uncertainty using descriptor similarity scores.  
> If the visual system is uncertain, allow language information to help.  
> If the visual system is confident, trust vision more.

---

## Current Research Checkpoint

This repository currently focuses on three reproducibility tasks:

1. Running visual-only baselines on the AmsterTime dataset.
2. Checking whether descriptor-derived similarity matrices are sorted before computing uncertainty metrics.
3. Implementing uncertainty metrics for future confidence-gated visual-language fusion.

---

## Dataset

The current dataset is **AmsterTime**.

AmsterTime contains matched old and new images of the same locations. In this checkpoint:

- `old` images are used as the database images.
- `new` images are used as the query images.
- The Kaggle version of AmsterTime was used because the original 4TU downloader returned an HTML page instead of a valid zip file during reproduction.
- The Kaggle filenames were reformatted with index-based artificial UTM labels so that `auto_VPR` could compute recall metrics.

Example reformatted filename:

```text
0001@1000.0@0.0@.jpg
