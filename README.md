# VPR Language Fusion

Confidence-gated language fusion for Visual Place Recognition (VPR).

This repository contains the code, result tables, and reproducibility scripts for a training-free language-fusion method for VPR. The project studies whether natural-language scene descriptions can improve image-based place recognition when combined with visual similarity scores.

The main finding is that fixed-weight language fusion is not always safe. Language can help in some datasets, but it can also degrade retrieval when generated descriptions are noisy or generic. The proposed method uses confidence-gated top-10 reranking to apply language selectively.

---

## 1. Project Summary

Visual Place Recognition systems usually retrieve places using visual descriptors only. Recent language-augmented VPR methods suggest that image descriptions can provide additional semantic information. However, using language with a fixed fusion weight can be risky because generated descriptions may be generic, noisy, or incorrect.

This project evaluates three settings:

1. **B1 Visual-only baseline**  
   EigenPlaces visual retrieval using ResNet50 descriptors.

2. **B2 Fixed-fusion baseline**  
   Fixed-weight fusion between EigenPlaces visual similarity and BGE-large language similarity using alpha = 0.5.

3. **Confidence-gated fusion**  
   A training-free method that re-ranks the top-10 visual candidates using a dynamic fusion weight. The gate uses Statistical Uncertainty (SU) from the visual similarity distribution and suppresses language when language similarities are not discriminative.

The primary method reported in the paper is:

```text
Confidence-Gated-Top10
```

Full-database fusion is reported as an ablation.

---

## 2. Main Claim

The main contribution is not that language always improves VPR. Instead, the results show that:

```text
Fixed-weight language fusion can be harmful when language descriptions are noisy.
Confidence-gated fusion provides a safer candidate-reranking strategy by using language selectively.
```

This is most visible on Nordland-clean, where fixed fusion causes a large R@1 degradation, while the proposed confidence-gated top-10 method recovers the loss.

---

## 3. Datasets

| Dataset | Description | Query / Database Size |
|---|---|---:|
| AmsterTime | Historical-to-modern city image matching | 1231 queries / 1231 database |
| MSLS-val | Mapillary Street-Level Sequences validation subset | 740 queries / 18871 database |
| Nordland-clean | Aligned summer-winter railway subset | 400 queries / 400 database |

### Nordland-clean note

The original equal-timestamp Nordland extraction was not valid because the summer and winter videos had non-linear temporal drift. A cleaned 400-pair aligned subset was created using:

```text
Summer database start index: 1900
Winter query start index:   2000
Number of pairs:            400
Synthetic coordinate spacing: 1 metre
Positive threshold:          25 metres = ±25 sampled frames
```

This makes the Nordland-clean evaluation reproducible and avoids reporting invalid equal-timestamp results.

---

## 4. Main Results

Primary result: **top-10 confidence-gated reranking**.

All results are reported as R@1 / R@5 / R@10.

| Dataset | B1 Visual Only | B2 Fixed Fusion | Ours Top-10 | Main Observation |
|---|---:|---:|---:|---|
| AmsterTime | 43.4 / 63.4 / 70.2 | 43.6 / 65.8 / 73.4 | 43.2 / 63.8 / 70.2 | Within noise of visual baseline |
| MSLS-val | 84.9 / 90.9 / 92.8 | 84.5 / 91.5 / 93.2 | 85.1 / 91.1 / 92.8 | Best R@1 |
| Nordland-clean | 52.8 / 76.2 / 87.8 | 45.8 / 75.0 / 83.8 | 53.0 / 76.0 / 87.8 | Recovers fixed-fusion degradation |

### Key Nordland-clean finding

```text
B1 visual-only R@1:       52.8
B2 fixed fusion R@1:      45.8
Ours top-10 R@1:          53.0
```

Fixed fusion drops R@1 by **7.0 percentage points** compared with the visual-only baseline. The confidence-gated method recovers this degradation and improves by **+7.2 percentage points** compared with fixed fusion.

### AmsterTime significance check

On AmsterTime, the proposed top-10 result is 3 queries lower than B1:

```text
B1 correct:          535 / 1231
Ours top-10 correct: 532 / 1231
Difference:          -3 queries
Bootstrap 95% CI:    [-1.38, +0.89] percentage points
```

Because the confidence interval includes zero, this difference is treated as statistically indistinguishable from the visual-only baseline.

---

## 5. Ablation Results

### 5.1 Full-database fusion variant

| Dataset | Ours Full-Database | Notes |
|---|---:|---|
| AmsterTime | 43.5 / 63.8 / 70.5 | Exactly matches B1 R@1 query count |
| MSLS-val | 85.0 / 90.9 / 92.7 | Slight R@1 improvement over B1 |
| Nordland-clean | 52.2 / 74.5 / 85.0 | Recovers most of B2 degradation |

Full-database fusion is reported as an ablation. The main method is top-10 reranking because the intended design is to retrieve visual candidates first, then selectively use language to re-rank them.

### 5.2 MSLS-val tau ablation

| Tau | Top-10 R@1 | Top-10 R@5 | Top-10 R@10 | Mean Alpha |
|---:|---:|---:|---:|---:|
| 0.5 | 85.1 | 91.1 | 92.8 | 0.893 |
| 1.0 | 85.1 | 91.1 | 92.8 | 0.885 |
| 2.0 | 85.0 | 91.1 | 92.8 | 0.875 |

The tau ablation shows that the method is stable across tau values. The default setting used in the main experiments is:

```text
tau = 1.0
language threshold = 0.05
top-K = 10
```

### 5.3 Prompted Nordland captions

A railway-specific prompted-caption experiment was also tested on Nordland-clean. It removed obvious caption hallucinations and improved R@5, but it reduced R@1 compared with the original gated result.

```text
Prompted top-10 result: 51.5 / 78.0 / 87.8
Original top-10 result: 53.0 / 76.0 / 87.8
```

Therefore, the prompted-caption experiment is treated as exploratory and is not used as the primary result.

---

## 6. Repository Structure

```text
vpr-language-fusion/
│
├── src/
│   └── su_metric.py
│
├── scripts/
│   ├── download_amstertime.sh
│   ├── check_similarity_sorting.py
│   ├── session_init.py
│   ├── run_b1_baseline.py
│   ├── run_b2_baseline.py
│   ├── run_method.py
│   ├── run_tau_ablation.py
│   └── create_nordland_clean.py
│
├── results/
│   ├── baseline_results.csv
│   ├── amstertime_paired_bootstrap_r1.csv
│   ├── msls_val_tau_ablation.csv
│   ├── amstertime_cosplace_sorting_check.txt
│   └── amstertime_eigenplaces_b1_sorting_check.txt
│
├── figures/
│   └── su_calibration_msls.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 7. Reproducibility

### 7.1 Environment setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For a fresh Colab session, run:

```bash
python scripts/session_init.py
```

This clones required repositories, prepares Google Drive folders, and checks whether GPU is available.

### 7.2 Download or prepare datasets

AmsterTime can be downloaded using:

```bash
bash scripts/download_amstertime.sh
```

MSLS-val requires Mapillary access and must be prepared according to the project notes.

Nordland-clean is a custom aligned subset and can be recreated using:

```bash
python scripts/create_nordland_clean.py
```

### 7.3 Run visual-only B1 baseline

```bash
python scripts/run_b1_baseline.py --dataset amstertime
python scripts/run_b1_baseline.py --dataset msls_val
python scripts/run_b1_baseline.py --dataset nordland_clean
```

Expected R@1 / R@5 / R@10:

```text
amstertime:      43.4 / 63.4 / 70.2
msls_val:        84.9 / 90.9 / 92.8
nordland_clean:  52.8 / 76.2 / 87.8
```

### 7.4 Run fixed-fusion B2 baseline

```bash
python scripts/run_b2_baseline.py --dataset amstertime
python scripts/run_b2_baseline.py --dataset msls_val
python scripts/run_b2_baseline.py --dataset nordland_clean
```

### 7.5 Run confidence-gated fusion

```bash
python scripts/run_method.py --dataset amstertime --tau 1.0
python scripts/run_method.py --dataset msls_val --tau 1.0
python scripts/run_method.py --dataset nordland_clean --tau 1.0
```

Expected primary R@1 results:

```text
amstertime:      43.2
msls_val:        85.1
nordland_clean:  53.0
```

### 7.6 Run tau ablation

```bash
python scripts/run_tau_ablation.py --dataset msls_val
```

---

## 8. Saved Matrices and Evidence

Large `.npy` files are not stored in this repository. They are saved in Google Drive under:

```text
/content/drive/MyDrive/vpr_research/embeddings/
/content/drive/MyDrive/vpr_research/results/
/content/drive/MyDrive/vpr_research/figures/
```

The expected saved matrices are:

```text
amstertime_visual_sim_matrix.npy
amstertime_lang_sim_matrix.npy
amstertime_positive_matrix.npy

msls_val_visual_sim_matrix.npy
msls_val_lang_sim_matrix.npy
msls_val_positive_matrix.npy

nordland_clean_visual_sim_matrix.npy
nordland_clean_lang_sim_matrix.npy
nordland_clean_positive_matrix.npy
```

Additional evidence files include:

```text
amstertime_paired_bootstrap_r1.csv
msls_val_tau_ablation.csv
su_calibration_msls.png
amstertime_cosplace_sorting_check.txt
amstertime_eigenplaces_b1_sorting_check.txt
```

---

## 9. Method Details

The confidence-gated method uses the top-10 visual candidates. For each query:

1. Retrieve top-10 candidates using EigenPlaces visual similarity.
2. Sort visual scores in descending order.
3. Compute Statistical Uncertainty (SU) from the visual similarity distribution.
4. Convert normalised SU into a visual weight alpha using a sigmoid gate.
5. Check whether language similarities are discriminative.
6. If language is discriminative, fuse visual and language scores.
7. If language is not discriminative, suppress language and keep visual-only scores.
8. Re-rank the top-10 candidates.

The fusion equation is:

```text
fused_score = alpha × visual_similarity + (1 − alpha) × language_similarity
```

where:

```text
alpha close to 1 = trust vision more
alpha close to 0 = trust language more
```

---

## 10. Notes for Reviewers

This repository separates three types of evidence:

1. **Clean reproducibility scripts**  
   Stored in `scripts/`.

2. **Final result tables and statistical checks**  
   Stored in `results/`.

3. **Raw exploratory coding history**  
   Stored outside the main repository as archived notebook evidence. The exploratory notebook includes debugging, failed attempts, and intermediate trials, and is not intended to be run from top to bottom.

The official reproducibility path is through the scripts, saved matrices, result CSVs, and README instructions.

---

## 11. Citation

Citation details will be added after manuscript submission.
