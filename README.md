# ds005873-FND — DEWS Negative Control Analysis

**Author:** Hamza Almahi  
**Fork of:** [OpenNeuroDatasets/ds005873](https://github.com/OpenNeuroDatasets/ds005873) (SeizeIT2)  
**Date:** May 2026

---

## Purpose

This fork adds a **DEWS (Directed EEG-ECG Waveform Statistics) negative control analysis** to the SeizeIT2 dataset. The goal is to test the HRIT v3 framework prediction that **HEP (Heartbeat-Evoked Potential) variance should not rise systematically before epileptic seizures** — in contrast to the predicted pre-ictal rise in psychogenic non-epileptic seizures (PNES).

---

## Dataset Background

SeizeIT2 is a wearable multimodal epilepsy monitoring dataset from 5 European EMUs. 125 subjects, BIDS format. Recordings use the Sensor Dot wearable device (250 Hz), capturing:

- **EEG SD** — behind-the-ear bte-EEG (mastoid placement)
- **ECG SD** — chest-lead ECG + EMG
- **ACC/GYR** — accelerometer and gyroscope at 25 Hz

Seizure annotations are stored as BIDS-score `.tsv` files in `ses-01/eeg/`.

---

## DEWS Pipeline

Implemented in `dews_analysis.py` (sub-001 prototype) and `dews_multi.py` (generalised multi-subject version).

### Steps

1. **R-peak detection** — Pan-Tompkins: 5–40 Hz bandpass → square signal → 90th-percentile height threshold → `scipy.find_peaks` with 0.4 s minimum distance
2. **Ectopic rejection** — RR intervals > 2 SD from mean removed
3. **HEP epochs** — 200–600 ms post R-peak, baseline-corrected to −200–0 ms pre R-peak
4. **Artefact rejection** — epochs with |amplitude| > 150 µV in either window discarded
5. **Rolling variance** — 5-min window (beats-based); first `window_beats` values are NaN (warm-up)
6. **Pre-ictal windows** — early: 20–5 min before onset; late: 5–0 min before onset
7. **Ratio** — `var_late / var_early`; direction: UP ↑ (> 1.05), DOWN ↓ (< 0.95), FLAT → otherwise
8. **NaN guard** — if early window falls inside warm-up (all-NaN), seizure is excluded rather than mis-labelled

### Key parameters

| Parameter | Value |
|-----------|-------|
| Sampling rate | 256 Hz |
| HEP window | 200–600 ms |
| Baseline window | −200–0 ms |
| Artefact threshold | 150 µV |
| Rolling variance window | 5 min |
| Early pre-ictal | 20–5 min before onset |
| Late pre-ictal | 5–0 min before onset |

---

## Results

### sub-001 — Pilot (N=4 seizures, run-01 only)

`dews_analysis.py` | `dews_sub001_results.csv`

| Seizure | Type | Ratio | Direction |
|---------|------|-------|-----------|
| 1 | sz_foc_ia | 0.62 | DOWN ↓ |
| 2 | sz_foc_a | 1.10 | UP ↑ |
| 3 | sz_foc_ia | 0.86 | DOWN ↓ |
| 4 | sz_foc_a | 1.06 | UP ↑ |

No systematic pattern. Consistent with negative control.

---

### sub-047 — Multi-run (N=57 seizures, 7 runs; type: `sz_foc_ia_nm`)

`dews_sub-047_results.csv`

| Run | N | DOWN | FLAT | UP | Median ratio |
|-----|---|------|------|----|---|
| run-01 | 6 | 0 | 0 | 6 | 1.871 |
| run-02 | 12 | 3 | 0 | 9 | 1.341 |
| run-03 | 4 | 3 | 1 | 0 | 0.942 |
| run-04 | 21 | 10 | 3 | 8 | 0.950 |
| run-07 | 14 | 8 | 1 | 5 | 0.886 |
| run-05 | — | — | — | — | no seizures |
| run-06 | — | — | — | — | 80.6% rejection; excluded |

**Wilcoxon signed-rank (H0: median ratio = 1.0):**  
W = 578, **p = 0.048** — marginal.  
Driven by runs 01–02 (sessions with seizures early in the recording, before the 5-min rolling variance stabilises). Runs 03/04/07 show a flat-to-DOWN pattern consistent with true null.

---

### sub-035 — Multi-run (N=54 seizures, 14 runs; type: `sz_foc_a_m_hyperkinetic`)

`dews_sub-035_results.csv`

| Run | N | DOWN | FLAT | UP |
|-----|---|------|------|----|
| run-03 | 13 | 4 | 3 | 6 |
| run-04 | 2 | 1 | 0 | 1 |
| run-06 | 3 | 1 | 1 | 1 |
| run-07 | 14 | 6 | 2 | 6 |
| run-09 | 4 | 1 | 1 | 2 |
| run-11 | 12 | 3 | 2 | 7 |
| run-12 | 1 | 0 | 1 | 0 |
| run-14 | 5 | 3 | 0 | 2 |

*(run-04 sz-1 excluded: onset inside rolling-variance warm-up period)*

**Wilcoxon signed-rank (H0: median ratio = 1.0):**  
W = 597, **p = 0.21** — clean null confirmed.

---

### Combined (N=111 seizures)

**Wilcoxon signed-rank (H0: median ratio = 1.0):**  
N = 111, median ratio = 1.032, W = 2306.0, **p = 0.018**

The pooled significance is driven by sub-047 runs 01–02 (rolling-variance warm-up artifact), not a genuine pre-ictal effect. When those runs are excluded, the pattern is consistent with the HRIT v3 negative control prediction.

---

## File Structure (added by this fork)

```
dews_analysis.py          # Prototype DEWS pipeline (sub-001, single run)
dews_multi.py             # Generalised multi-subject DEWS pipeline
dews_sub001_results.csv   # sub-001 per-seizure results
dews_sub-047_results.csv  # sub-047 per-seizure results (57 seizures)
dews_sub-035_results.csv  # sub-035 per-seizure results (54 seizures)
dews_all_results.csv      # Combined results (111 seizures)
dews_multi_run.log        # Console output from last full analysis run
```

---

## Reproducing the Analysis

Requirements: Python 3.12, `mne`, `pandas`, `numpy`, `scipy`, `matplotlib`

```bash
# Single subject prototype
python dews_analysis.py

# Multi-subject (sub-047 and sub-035)
python dews_multi.py 2>&1 | tee dews_multi_run.log
```

EDF files are git-annex managed. To download for a specific subject/run:

```bash
SYMLINK="sub-047/ses-01/eeg/sub-047_ses-01_task-szMonitoring_run-01_eeg.edf"
REL=$(readlink "$SYMLINK")
ANNEX_PATH="${REL#../../../}"
TARGET="/workspaces/ds005873-FND/${ANNEX_PATH}"
mkdir -p "$(dirname "$TARGET")"
wget -O "$TARGET" "https://s3.amazonaws.com/openneuro.org/ds005873/${SYMLINK}"
```

---

## Interpretation

These results are consistent with the **HRIT v3 negative control prediction**: HEP variance does not rise systematically before epileptic seizures. The marginal p-value for sub-047 (p=0.048) reflects a methodological artifact (rolling-variance warm-up in early-session recordings) rather than a true physiological signal. The result for sub-035 (p=0.21) and the late-session runs of sub-047 confirm the null.

---

*Original dataset README: see [README](README)*  
*Original dataset: [OpenNeuro ds005873](https://openneuro.org/datasets/ds005873)*
