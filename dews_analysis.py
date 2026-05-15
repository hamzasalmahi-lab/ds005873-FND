"""
DEWS Analysis — SeizeIT2 (sub-001)
Negative control arm: HEP variance before epileptic seizures
HRIT v3 prediction: variance should NOT rise pre-ictally in epilepsy

Method:
- R-peak detection: Pan-Tompkins (bandpass 5-40 Hz, squared signal)
- HEP epoch: 200-600 ms post R-peak
- Baseline correction: -200 to 0 ms pre R-peak subtracted per epoch
- Artefact rejection: |amplitude| > 150 µV in epoch OR baseline window
- Rolling variance: 5-min window (beats-based)
- Summary: early (20-5 min pre-onset) vs late (5-0 min pre-onset) variance
"""

import mne
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# ── CONFIG ───────────────────────────────────────────────────────────────────
EEG_PATH = Path("sub-001/ses-01/eeg"
                "/sub-001_ses-01_task-szMonitoring_run-01_eeg.edf")
ECG_PATH = Path("sub-001/ses-01/ecg"
                "/sub-001_ses-01_task-szMonitoring_run-01_ecg.edf")
ECG_CH        = "ECG SD"
EEG_CH_IDX    = 0          # BTEleft SD
SFREQ         = 256

HEP_START_MS  = 200        # epoch start post R-peak
HEP_END_MS    = 600        # epoch end post R-peak
BASE_START_MS = -200       # baseline start pre R-peak
BASE_END_MS   = 0          # baseline end pre R-peak
REJECT_UV     = 150        # artefact threshold (µV)

WINDOW_MIN    = 5          # rolling variance window
PRE_ICTAL_MIN = 30         # window to plot before seizure
EARLY_MIN     = (20, 5)    # early pre-ictal window for summary
LATE_MIN      = (5, 0)     # late pre-ictal window for summary

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading data...")
raw_ecg = mne.io.read_raw_edf(ECG_PATH, preload=True, verbose=False)
raw_eeg = mne.io.read_raw_edf(EEG_PATH, preload=True, verbose=False)
ecg     = raw_ecg[ECG_CH][0][0]
eeg     = raw_eeg[EEG_CH_IDX][0][0]

print(f"  EEG channel: {raw_eeg.ch_names[EEG_CH_IDX]}")
print(f"  ECG channel: {ECG_CH}")
print(f"  Duration: {raw_eeg.times[-1]/3600:.2f} h")

# ── SEIZURES ─────────────────────────────────────────────────────────────────
all_tsv = sorted(Path("sub-001").rglob("*.tsv"))
dfs = []
for f in all_tsv:
    df = pd.read_csv(f, sep='\t')
    if 'eventType' in df.columns:
        dfs.append(df)
annot    = pd.concat(dfs, ignore_index=True)
seizures = (annot[annot['eventType'].str.startswith('sz', na=False)]
            .sort_values('onset')
            .reset_index(drop=True))
print(f"\nSeizures: {len(seizures)}")

# ── R-PEAK DETECTION ─────────────────────────────────────────────────────────
print("\nDetecting R-peaks...")
ecg_filt = mne.filter.filter_data(
    ecg, SFREQ, l_freq=5, h_freq=40, verbose=False)
ecg_sq   = ecg_filt ** 2
peaks, _ = find_peaks(
    ecg_sq,
    distance=int(0.4 * SFREQ),
    height=np.percentile(ecg_sq, 90))

rr_intervals = np.diff(peaks) / SFREQ
hr_mean      = 60 / np.mean(rr_intervals)
hr_std       = 60 * np.std(rr_intervals) / np.mean(rr_intervals) ** 2
print(f"  R-peaks: {len(peaks)}")
print(f"  Mean HR: {hr_mean:.1f} ± {hr_std:.1f} bpm")

# Reject ectopic beats (RR interval > 2 SD from mean)
rr_mean = np.mean(rr_intervals)
rr_std  = np.std(rr_intervals)
valid_peaks = [peaks[0]]
for i, p in enumerate(peaks[1:], 1):
    rr = (p - peaks[i-1]) / SFREQ
    if abs(rr - rr_mean) < 2 * rr_std:
        valid_peaks.append(p)

n_ectopic = len(peaks) - len(valid_peaks)
print(f"  Ectopic beats removed: {n_ectopic}")
peaks = np.array(valid_peaks)

# ── HEP EXTRACTION ───────────────────────────────────────────────────────────
hep_s  = int(HEP_START_MS  / 1000 * SFREQ)
hep_e  = int(HEP_END_MS    / 1000 * SFREQ)
base_s = int(BASE_START_MS / 1000 * SFREQ)   # negative
base_e = int(BASE_END_MS   / 1000 * SFREQ)   # 0

print(f"\nExtracting HEP epochs ({HEP_START_MS}-{HEP_END_MS} ms)...")

hep_times  = []
hep_amps   = []
n_rejected = 0

for p in peaks:
    b_start = p + base_s     # baseline window start (p - 200 samples)
    b_end   = p + base_e     # baseline window end   (p)
    e_start = p + hep_s      # epoch start
    e_end   = p + hep_e      # epoch end

    # Bounds check
    if b_start < 0 or e_end >= len(eeg):
        n_rejected += 1
        continue

    baseline_data = eeg[b_start:b_end]
    epoch_data    = eeg[e_start:e_end]

    # Artefact rejection on both windows
    if (np.max(np.abs(baseline_data)) * 1e6 > REJECT_UV or
            np.max(np.abs(epoch_data)) * 1e6 > REJECT_UV):
        n_rejected += 1
        continue

    # Baseline correction
    epoch_bc = epoch_data - np.mean(baseline_data)

    hep_times.append(p / SFREQ)
    hep_amps.append(np.mean(epoch_bc) * 1e6)

hep_times = np.array(hep_times)
hep_amps  = np.array(hep_amps)
print(f"  Valid epochs: {len(hep_times)}")
print(f"  Rejected:     {n_rejected} "
      f"({100*n_rejected/(len(hep_times)+n_rejected):.1f}%)")

# ── ROLLING VARIANCE ─────────────────────────────────────────────────────────
beats_per_sec = len(hep_times) / (hep_times[-1] - hep_times[0])
window_beats  = int(WINDOW_MIN * 60 * beats_per_sec)
print(f"\nRolling variance window: {window_beats} beats (~{WINDOW_MIN} min)")

rolling_var  = np.full(len(hep_amps), np.nan)
rolling_mean = np.full(len(hep_amps), np.nan)

for i in range(window_beats, len(hep_amps)):
    w = hep_amps[i - window_beats:i]
    rolling_var[i]  = np.var(w)
    rolling_mean[i] = np.mean(w)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n── DEWS SUMMARY ──")
print(f"{'#':<4} {'Type':<30} {'Early var':>10} {'Late var':>10} "
      f"{'Ratio':>7} {'Dir':>6} {'n_early':>8} {'n_late':>7}")
print("-" * 85)

results = []
for si, (_, row) in enumerate(seizures.iterrows()):
    onset_s    = row['onset']
    early_s    = onset_s - EARLY_MIN[0] * 60
    early_e    = onset_s - EARLY_MIN[1] * 60
    late_s     = onset_s - LATE_MIN[0]  * 60
    late_e     = onset_s - LATE_MIN[1]  * 60

    m_early = (hep_times >= early_s) & (hep_times < early_e)
    m_late  = (hep_times >= late_s)  & (hep_times < late_e)

    if m_early.sum() > 5 and m_late.sum() > 5:
        var_e = np.nanmean(rolling_var[m_early])
        var_l = np.nanmean(rolling_var[m_late])
        ratio = var_l / var_e
        dirn  = "UP ↑" if ratio > 1.05 else ("DOWN ↓" if ratio < 0.95 else "FLAT →")
        results.append(dict(
            seizure=si+1, type=row['eventType'],
            var_early=var_e, var_late=var_l,
            ratio=ratio, direction=dirn,
            n_early=m_early.sum(), n_late=m_late.sum()))
        print(f"{si+1:<4} {row['eventType']:<30} {var_e:>10.3f} "
              f"{var_l:>10.3f} {ratio:>7.2f} {dirn:>6} "
              f"{m_early.sum():>8} {m_late.sum():>7}")
    else:
        print(f"{si+1:<4} {row['eventType']:<30} {'insufficient data':>40}")

# ── PLOTS ────────────────────────────────────────────────────────────────────
LGREY = '#94A3B8'; TEAL = '#028090'
AMBER = '#D97706'; RED  = '#B91C1C'; NAVY = '#0D1B2A'

fig, axes = plt.subplots(len(seizures), 2,
                         figsize=(16, 4.5 * len(seizures)))
if len(seizures) == 1:
    axes = axes.reshape(1, -1)

for si, (_, row) in enumerate(seizures.iterrows()):
    onset_s = row['onset']
    label   = row['eventType'].replace('sz_foc_', '')
    start_s = onset_s - PRE_ICTAL_MIN * 60

    mask = (hep_times >= start_s) & (hep_times <= onset_s + 120)
    t    = (hep_times[mask] - onset_s) / 60   # minutes from onset
    amp  = hep_amps[mask]
    var  = rolling_var[mask]
    mean = rolling_mean[mask]

    for ax_col, data, ylabel, color, title_sfx in [
        (0, amp,  "HEP amplitude (µV)", TEAL,  "HEP mean"),
        (1, var,  "Variance (µV²)",     AMBER, "HEP variance — DEWS"),
    ]:
        ax = axes[si, ax_col]
        if ax_col == 0:
            ax.plot(t, amp, lw=0.5, color=LGREY, alpha=0.5)
            ax.plot(t, mean, lw=2.0, color=color, label='Rolling mean')
        else:
            ax.plot(t, data, lw=2.0, color=color, label='Rolling variance')

        ax.axvspan(-EARLY_MIN[0], -EARLY_MIN[1],
                   alpha=0.08, color='blue', label='Early window')
        ax.axvspan(-LATE_MIN[0],  0,
                   alpha=0.08, color='red',  label='Late window')
        ax.axvline(0, color=RED, lw=1.5, ls='--', label='Seizure onset')
        ax.set_title(f"Seizure {si+1} — {label} — {title_sfx}", fontsize=10)
        ax.set_xlabel("Time to onset (min)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.suptitle(
    "sub-001 — DEWS negative control (epileptic seizures)\n"
    "HRIT v3 prediction: no pre-ictal variance rise in epilepsy",
    fontsize=12, color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig("dews_sub001_final.png", dpi=150, bbox_inches='tight')
print("\nFigure saved: dews_sub001_final.png")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("dews_sub001_results.csv", index=False)
print("Results saved: dews_sub001_results.csv")
print("\nDone.")
