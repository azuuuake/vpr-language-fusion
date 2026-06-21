# Confidence-Gated Language Fusion for Safer Visual Place Recognition

This project studies when natural-language scene descriptions help or harm Visual Place Recognition (VPR). The main finding is conservative: **language fusion should not be applied blindly**. Fixed-weight visual-language fusion can slightly help, do almost nothing, or substantially degrade retrieval depending on the dataset and description quality.

The proposed method is a **training-free confidence-gated top-10 reranking module**. It uses visual retrieval uncertainty and language-score discriminability to decide when language should influence the candidate ranking.

---

## 1. Quick Summary

### What is being tested?

We compare three main settings:

1. **B1 Visual-only baseline**
   EigenPlaces visual retrieval using ResNet50 descriptors.

2. **B2 Fixed-fusion baseline**
   Fixed fusion between EigenPlaces visual similarity and BGE-large language similarity using fixed `alpha = 0.5`.

3. **Confidence-Gated-Top10**
   The proposed training-free method. It first retrieves top-10 visual candidates, then selectively reranks them using language only when:

   * the visual retrieval distribution is uncertain, and
   * the language similarities are sufficiently discriminative.

### Main result

The strongest safety result is on **Nordland-clean**:

| Method          |   R@1 / R@5 / R@10 |
| --------------- | -----------------: |
| B1 Visual-only  | 52.8 / 76.2 / 87.8 |
| B2 Fixed fusion | 45.8 / 75.0 / 83.8 |
| Ours Top-10     | 53.0 / 76.0 / 87.8 |

Fixed fusion reduces R@1 by **7.0 percentage points** compared with visual-only retrieval. Confidence-gated top-10 reranking recovers this fixed-fusion degradation.

This is the central claim of the menuscript:

```text
The method is intended as a safer fusion strategy, not as a claim that language always improves VPR.
```

---

## 2. Evidence Map

| Paper claim                              | Repository evidence                                                                                                                           |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Main B1/B2/Ours results                  | `results/baseline_results.csv`, `results/gating_ablation_results.csv`                                                                         |
| Component ablation table                 | `results/gating_ablation_results.csv`, `results/gating_ablation_paper_table.csv`                                                              |
| AmsterTime bootstrap confidence interval | `results/amstertime_paired_bootstrap_r1.csv`                                                                                                  |
| MSLS-val tau stability                   | `results/msls_val_tau_ablation.csv`                                                                                                           |
| Nordland prompted-caption stress test    | `results/nordland_clean_prompted_summary_results.csv`                                                                                         |
| SU calibration figure                    | `figures/su_calibration_msls.png`                                                                                                             |
| Sorting requirement before RS/SD/SU      | `results/amstertime_cosplace_sorting_check.txt`, `results/amstertime_eigenplaces_b1_sorting_check.txt`, `scripts/check_similarity_sorting.py` |
| Nordland-clean construction              | `scripts/create_nordland_clean.py`                                                                                                            |
| Main method implementation               | `src/su_metric.py`, `scripts/run_method.py`                                                                                                   |

---

## 3. Method Overview

The method uses the top-10 visual candidates returned by EigenPlaces. For each query:

1. Retrieve top-10 candidates using visual similarity.
2. Compute visual uncertainty from the top-10 visual similarity distribution.
3. Convert normalized Statistical Uncertainty (SU) into a continuous visual weight.
4. Compute the standard deviation of top-10 language similarities.
5. Suppress language if the language scores are not discriminative.
6. If language is allowed, fuse visual and language scores.
7. Rerank only the top-10 visual candidates.

The fused score is:

```text
fused_score = alpha × visual_similarity + (1 − alpha) × language_similarity
```

where:

```text
alpha close to 1 = vision dominates
alpha lower than 1 = language contributes more
```

The main settings used in the paper are:

```text
top-K = 10
tau = 1.0
language threshold theta = 0.05
visual backbone = EigenPlaces ResNet50
text encoder = BGE-large
image resolution = 320 × 320
```

---

## 4. Datasets

| Dataset        | Purpose                               |        Query / Database Size | Description source          |
| -------------- | ------------------------------------- | ---------------------------: | --------------------------- |
| AmsterTime     | Historical-to-modern temporal drift   | 1231 queries / 1231 database | LaVPR-provided descriptions |
| MSLS-val       | Multi-condition urban retrieval       | 740 queries / 18871 database | LaVPR-provided descriptions |
| Nordland-clean | Seasonal summer-winter railway subset |   400 queries / 400 database | BLIP-base captions          |

### Nordland-clean construction

Nordland-clean is a controlled 400-pair aligned summer-winter subset. It was created because a naive equal-timestamp extraction was not reliable due to temporal drift between the seasonal videos.

The final subset uses:

```text
Summer database start index: 1900
Winter query start index:   2000
Number of pairs:            400
Synthetic coordinate spacing: 1 metre
Positive threshold:          25 metres = ±25 sampled frames
```

The synthetic coordinate spacing is used only to express the auto_VPR 25m evaluation threshold as a frame-index tolerance. It is not a physical GPS claim.

The subset can be recreated using:

```bash
python scripts/create_nordland_clean.py
```

---

## 5. Main Results

All results are reported as:

```text
R@1 / R@5 / R@10
```

| Dataset        |     B1 Visual-only |    B2 Fixed fusion |        Ours Top-10 | Main interpretation                                                   |
| -------------- | -----------------: | -----------------: | -----------------: | --------------------------------------------------------------------- |
| AmsterTime     | 43.5 / 63.4 / 70.2 | 43.6 / 65.8 / 73.4 | 43.2 / 63.8 / 70.2 | Statistically indistinguishable from B1                               |
| MSLS-val       | 84.9 / 90.9 / 92.8 | 84.5 / 91.5 / 93.2 | 85.1 / 91.1 / 92.8 | Highest nominal R@1 among tested methods, but only +2 queries over B1 |
| Nordland-clean | 52.8 / 76.2 / 87.8 | 45.8 / 75.0 / 83.8 | 53.0 / 76.0 / 87.8 | Recovers fixed-fusion degradation                                     |

### Conservative interpretation

The proposed method is **not** claimed to be a universal improvement over visual-only retrieval.

* On **MSLS-val**, the improvement over B1 is small: 85.1 vs 84.9 R@1, corresponding to two additional top-1 correct queries.
* On **AmsterTime**, the method is three top-1 queries below B1 and statistically indistinguishable from B1.
* On **Nordland-clean**, the main result is recovery from the fixed-fusion collapse, not a statistically meaningful improvement over visual-only retrieval.

The intended conclusion is:

```text
Fixed language fusion can be unsafe.
Confidence-gated fusion reduces this risk by applying language selectively.
```

---

## 6. AmsterTime Paired Bootstrap Check

On AmsterTime:

```text
B1 correct:          535 / 1231
Ours Top-10 correct: 532 / 1231
Difference:          -3 queries
Bootstrap 95% CI:    [-1.38, +0.89] percentage points
```

Because the confidence interval includes zero, the AmsterTime result is treated as statistically indistinguishable from the visual-only baseline.

Evidence file:

```text
results/amstertime_paired_bootstrap_r1.csv
```

---

## 7. Component Ablation

The component ablation tests whether both parts of the method are needed:

* **SU-only top-10:** continuous SU weighting without the language-discriminability pre-filter.
* **Pre-filter-only top-10:** language-discriminability pre-filter with fixed `alpha = 0.5` whenever language is allowed.
* **Margin-gated top-10:** visual top-1/top-2 margin instead of SU, with the same language-discriminability pre-filter.
* **Full method top-10:** SU-derived continuous weighting plus language-discriminability pre-filter.

| Method                |         AmsterTime |           MSLS-val |     Nordland-clean |
| --------------------- | -----------------: | -----------------: | -----------------: |
| B1 visual-only        | 43.5 / 63.4 / 70.2 | 84.9 / 90.9 / 92.8 | 52.8 / 76.2 / 87.8 |
| B2 fixed fusion       | 43.6 / 65.8 / 73.4 | 84.5 / 91.5 / 93.2 | 45.8 / 75.0 / 83.8 |
| SU-only top10         | 43.1 / 64.7 / 70.2 | 84.1 / 91.2 / 92.8 | 49.5 / 78.2 / 87.8 |
| Pre-filter-only top10 | 43.7 / 63.7 / 70.2 | 84.9 / 91.1 / 92.8 | 51.8 / 76.0 / 87.8 |
| Margin-gated top10    | 43.1 / 63.7 / 70.2 | 85.0 / 91.1 / 92.8 | 51.5 / 76.0 / 87.8 |
| Full method top10     | 43.2 / 63.8 / 70.2 | 85.1 / 91.1 / 92.8 | 53.0 / 76.0 / 87.8 |

Evidence files:

```text
results/gating_ablation_results.csv
results/gating_ablation_paper_table.csv
scripts/run_gating_ablation.py
```

To verify the saved ablation result table:

```bash
python scripts/run_gating_ablation.py
```

This script validates the saved ablation CSV against the values reported in the manuscript. It does not recompute the full ablation from raw matrices.

---

## 8. Full-Database Fusion Ablation

The main method reranks only the top-10 visual candidates. Full-database fusion is reported as an ablation.

| Dataset        |        Ours Top-10 | Ours Full-Database |
| -------------- | -----------------: | -----------------: |
| AmsterTime     | 43.2 / 63.8 / 70.2 | 43.5 / 63.8 / 70.5 |
| MSLS-val       | 85.1 / 91.1 / 92.8 | 85.0 / 90.9 / 92.7 |
| Nordland-clean | 53.0 / 76.0 / 87.8 | 52.2 / 74.5 / 85.0 |

The largest difference is on Nordland-clean, where full-database fusion performs below top-10 reranking. This is consistent with increased exposure to unreliable captions when language is applied beyond visually filtered candidates.

---

## 9. MSLS-val Tau Ablation

Tau controls the sharpness of the SU-derived sigmoid gate. The paper uses `tau = 1.0`.

| Tau |  R@1 |  R@5 | R@10 | Mean alpha |
| --: | ---: | ---: | ---: | ---------: |
| 0.5 | 85.1 | 91.1 | 92.8 |      0.893 |
| 1.0 | 85.1 | 91.1 | 92.8 |      0.885 |
| 2.0 | 85.0 | 91.1 | 92.8 |      0.875 |

The result is stable across the tested tau values.

Evidence file:

```text
results/msls_val_tau_ablation.csv
```

Run:

```bash
python scripts/run_tau_ablation.py --dataset msls_val
```

---

## 10. Nordland Prompted-Caption Stress Test

The main Nordland-clean experiment uses BLIP-base captions. A second prompted-caption stress test was conducted to check whether the fixed-fusion failure was specific to the original BLIP-base captions.

Prompted captions were generated offline using:

```text
Model: llava-hf/llava-1.5-7b-hf
Generation settings: max_new_tokens = 80, do_sample = False
Output file: nordland_clean_prompted_descriptions.csv
```

Prompt:

```text
Describe this railway scene for location recognition using only permanent features: track geometry, terrain shape, hills, slopes, cuttings, embankments, bridges, tunnels, poles, signs, fences, and vegetation. Output one concise sentence from left to right. Ignore all appearance changes including weather and season.
```

Results:

| Caption source |          B1 Visual |       Fixed fusion |       Gated Top-10 |
| -------------- | -----------------: | -----------------: | -----------------: |
| BLIP-base      | 52.8 / 76.2 / 87.8 | 45.8 / 75.0 / 83.8 | 53.0 / 76.0 / 87.8 |
| Prompted       | 52.8 / 76.2 / 87.8 | 45.5 / 76.8 / 87.8 | 51.5 / 78.0 / 87.8 |

The prompted-caption test shows that the fixed-fusion failure is not limited to the original BLIP-base captions. Fixed fusion still drops R@1 from 52.8 to 45.5. The gated method recovers much of this loss, reaching 51.5, but remains below the visual-only baseline in this prompted setting.

This supports the conservative interpretation:

```text
Gating reduces harmful fusion failures, but it cannot guarantee improvement when the language signal remains semantically unreliable.
```

Evidence file:

```text
results/nordland_clean_prompted_summary_results.csv
```

---

## 11. SU Calibration Evidence

The method uses Statistical Uncertainty (SU) as a visual confidence proxy. On MSLS-val, queries were divided into ten SU deciles.

Key result:

```text
Most confident decile:
mean SU = 0.44
visual-only R@1 = 1.000

Most uncertain decile:
mean SU = 0.93
visual-only R@1 = 0.392

Separation:
60.8 percentage points
```

This supports the use of SU as a training-free visual confidence proxy.

Supporting figure:

```text
figures/su_calibration_msls.png
```

---

## 12. Repository Structure

```text
vpr-language-fusion/
│
├── src/
│   └── su_metric.py
│
├── scripts/
│   ├── session_init.py
│   ├── download_amstertime.sh
│   ├── check_similarity_sorting.py
│   ├── create_nordland_clean.py
│   ├── run_b1_baseline.py
│   ├── run_b2_baseline.py
│   ├── run_method.py
│   ├── run_tau_ablation.py
│   └── run_gating_ablation.py
│
├── results/
│   ├── baseline_results.csv
│   ├── gating_ablation_results.csv
│   ├── gating_ablation_paper_table.csv
│   ├── amstertime_paired_bootstrap_r1.csv
│   ├── msls_val_tau_ablation.csv
│   ├── nordland_clean_prompted_summary_results.csv
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

## 13. Reproducibility Instructions

### 13.1 Environment setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For a fresh Colab session:

```bash
python scripts/session_init.py
```

This clones the required repositories, prepares Google Drive folders, and checks the available device.

### 13.2 Download or prepare datasets

Download AmsterTime:

```bash
bash scripts/download_amstertime.sh
```

Prepare Nordland-clean:

```bash
python scripts/create_nordland_clean.py
```

MSLS-val requires Mapillary access and must be prepared according to the dataset access requirements.

### 13.3 Run B1 visual-only baseline

```bash
python scripts/run_b1_baseline.py --dataset amstertime
python scripts/run_b1_baseline.py --dataset msls_val
python scripts/run_b1_baseline.py --dataset nordland_clean
```

Expected R@1 / R@5 / R@10:

```text
amstertime:      43.5 / 63.4 / 70.2
msls_val:        84.9 / 90.9 / 92.8
nordland_clean:  52.8 / 76.2 / 87.8
```

### 13.4 Run B2 fixed-fusion baseline

```bash
python scripts/run_b2_baseline.py --dataset amstertime
python scripts/run_b2_baseline.py --dataset msls_val
python scripts/run_b2_baseline.py --dataset nordland_clean
```

Expected R@1 / R@5 / R@10:

```text
amstertime:      43.6 / 65.8 / 73.4
msls_val:        84.5 / 91.5 / 93.2
nordland_clean:  45.8 / 75.0 / 83.8
```

### 13.5 Run Confidence-Gated-Top10

```bash
python scripts/run_method.py --dataset amstertime --tau 1.0
python scripts/run_method.py --dataset msls_val --tau 1.0
python scripts/run_method.py --dataset nordland_clean --tau 1.0
```

Expected R@1 / R@5 / R@10:

```text
amstertime:      43.2 / 63.8 / 70.2
msls_val:        85.1 / 91.1 / 92.8
nordland_clean:  53.0 / 76.0 / 87.8
```

### 13.6 Run tau ablation

```bash
python scripts/run_tau_ablation.py --dataset msls_val
```

### 13.7 Verify component ablation table

```bash
python scripts/run_gating_ablation.py
```

This checks `results/gating_ablation_results.csv` against the component-ablation values reported in the manuscript.

---

## 14. Saved Matrices and Large Files

Large `.npy` similarity and positive matrices are not stored in this GitHub repository. They are expected under:

```text
/content/drive/MyDrive/vpr_research/embeddings/
```

Expected matrix files:

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

The repository keeps lightweight result evidence in `results/`, while full matrix-based reruns require the saved `.npy` files.

---

## 15. Sorting Check for SU

The descriptor-derived similarity matrix should not be assumed to be sorted by retrieval score. Its columns follow database image order unless explicitly sorted.

Before computing RS, SD, and SU, the score matrix must be sorted in descending order per query.

Evidence files:

```text
results/amstertime_cosplace_sorting_check.txt
results/amstertime_eigenplaces_b1_sorting_check.txt
scripts/check_similarity_sorting.py
```

The core rule is:

```python
sort_scores_desc(sim_matrix)
```

must be applied before uncertainty calculations.

---

## 16. Known Limitations

This repository and method have the following limitations:

1. **Language quality matters.**
   The language-discriminability pre-filter checks whether language similarities vary across candidates, but it does not verify whether captions are semantically correct.

2. **Nordland-clean is a 400-pair subset.**
   It was carefully aligned, but larger seasonal subsets would strengthen future evaluation.

3. **Large matrices are not included in GitHub.**
   The repository contains scripts and result evidence, while complete matrix-based reruns require the saved `.npy` files.

4. **MSLS-val access is controlled.**
   Recreating MSLS-val results requires dataset access through the appropriate source.

5. **The method is not claimed to be SOTA.**
   The claim is about safer language fusion behaviour under noisy descriptions, not about achieving state-of-the-art VPR performance.

---

## 17. Notes for Readers

This repository separates three types of material:

1. **Core method code**
   `src/su_metric.py`

2. **Reproducibility and verification scripts**
   `scripts/`

3. **Final result evidence**
   `results/` and `figures/`

The quickest verification path is:

```bash
python scripts/run_gating_ablation.py
```

Then inspect:

```text
results/gating_ablation_results.csv
results/baseline_results.csv
results/amstertime_paired_bootstrap_r1.csv
results/nordland_clean_prompted_summary_results.csv
figures/su_calibration_msls.png
```

The key result to verify first is the Nordland-clean fixed-fusion degradation:

```text
B1 Visual-only: 52.8 R@1
B2 Fixed fusion: 45.8 R@1
Ours Top-10:    53.0 R@1
```

This is the main evidence that fixed language fusion can be unsafe and that confidence-gated top-10 reranking can recover the fixed-fusion collapse.
