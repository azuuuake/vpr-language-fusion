# VPR Language Fusion

Confidence-gated language fusion for Visual Place Recognition (VPR).

This repository contains the code, result tables, and reproducibility scripts for a training-free language-fusion method for VPR. The project investigates whether natural-language scene descriptions can improve image-based place recognition when combined with visual similarity scores.

The key finding is that **fixed-weight language fusion is not always safe**. Language can help in some datasets, but it can also degrade retrieval when generated descriptions are noisy or hallucinated. The proposed method uses a confidence gate to apply language selectively, based on visual retrieval uncertainty and language-score discriminability.

---

## 1. Project Summary

Visual Place Recognition systems usually retrieve places using visual descriptors only. Recent language-augmented VPR methods suggest that image descriptions can provide additional semantic information. However, applying language with a fixed fusion weight can be risky because generated descriptions may be generic, noisy, or wrong.

This project evaluates three settings:

1. **B1 Visual-only baseline**  
   EigenPlaces visual retrieval using ResNet50 descriptors.

2. **B2 Fixed-fusion baseline**  
   Fixed-weight fusion between EigenPlaces visual similarity and BGE-large language similarity using alpha = 0.5.

3. **Confidence-gated fusion**  
   A training-free method that re-ranks the top-10 visual candidates using a dynamic fusion weight. The gate uses Statistical Uncertainty (SU) from the visual similarity distribution and suppresses language when language similarities are not discriminative.

The primary method is:

```text
Confidence-Gated-Top10
