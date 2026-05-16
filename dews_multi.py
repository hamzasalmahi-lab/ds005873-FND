"""
DEWS Multi-subject Analysis — SeizeIT2
Negative control arm: HEP variance before epileptic seizures
HRIT v3 prediction: variance should NOT rise pre-ictally in epilepsy

Usage:
    python dews_multi.py            # runs sub-047 and sub-035
    python dews_multi.py sub-001    # specific subject (all runs)

Each call to run_dews() processes one subject/run pair and returns a
DataFrame of per-seizure DEWS results.
"""

import sys
import mne
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# ── CONSTANTS ────────────────────────────────────────────────────────────────
SFREQ         = 256        # SeizeIT2 sampling rate (Hz)

HEP_START_MS  = 200        # epoch start post R-peak
HEP_END_MS    = 600        # epoch end post R-peak
BASE_START_MS = -200       # baseline start pre R-peak
BASE_END_MS   = 0          # baseline end pre R-peak
REJECT_UV     = 150        # artefact threshold (µV)

WINDOW_MIN    = 5          # rolling variance window
PRE_ICTAL_MIN = 30         # window to plot before seizure
EARLY_MIN     = (20, 5)    # early pre-ictal window for summary
LATE_MIN      = (5, 0)     # late pre-ictal window for summary

# Plot colours
LGREY = '#94A3B8'
TEAL  = '#028090'
AMBER = '#D97706'
RED   = '#B91C1C'
NAVY  = '#0D1B2A'


# ── MAIN FUNCTION ────────────────────────────────────────────────────────────

def run_dews(sub_id: str, run_id: str,
             base_path: Path = Path(".")) -> "pd.DataFrame | None":
    """
    Run DEWS analysis for one subject/run.
    Returns results DataFrame or None if no seizures found.

    Parameters
    ----------
    sub_id    : BIDS subject label, e.g. 'sub-047'
    run_id    : BIDS run label,     e.g. 'run-02'
    base_path : root of the BIDS dataset (default: current directory)

    Returns
    -------
    pd.DataFrame with columns:
        sub, run, seizure, type, var_early, var_late,
        ratio, direction, n_early, n_late
    or None if the run has no seizures / missing files.
    """
    # ── paths ────────────────────────────────────────────────────────────────
    stem     = f"{sub_id}_ses-01_task-szMonitoring_{run_id}"
    eeg_path = base_path / sub_id / "ses-01" / "eeg" / f"{stem}_eeg.edf"
    ecg_path = base_path / sub_id / "ses-01" / "ecg" / f"{stem}_ecg.edf"
    tsv_path = base_path / sub_id / "ses-01" / "eeg" / f"{stem}_events.tsv"

    tag = f"{sub_id}/{run_id}"

    for p, label in [(eeg_path, "EEG"), (ecg_path, "ECG"), (tsv_path, "TSV")]:
        if not p.exists():
            print(f"  [SKIP] {tag}: missing {label} ({p.name})")
            return None

    # ── seizures ─────────────────────────────────────────────────────────────
    annot = pd.read_csv(tsv_path, sep='\t')
    if 'eventType' not in annot.columns:
        print(f"  [SKIP] {tag}: no eventType column in TSV")
        return None

    seizures = (annot[annot['eventType'].str.startswith('sz', na=False)]
                .sort_values('onset')
                .reset_index(drop=True))

    if len(seizures) == 0:
        print(f"  [SKIP] {tag}: no seizures")
        return None

    print(f"\n── {tag}: {len(seizures)} seizure(s) ──")

    # ── load ─────────────────────────────────────────────────────────────────
    raw_ecg = mne.io.read_raw_edf(ecg_path, preload=True, verbose=False)
    raw_eeg = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)

    # auto-detect channels: first ECG channel, first EEG channel
    ecg_ch  = next(
        (ch for ch in raw_ecg.ch_names if 'ecg' in ch.lower()),
        raw_ecg.ch_names[0])
    eeg_ch  = raw_eeg.ch_names[0]

    ecg = raw_ecg[ecg_ch][0][0]
    eeg = raw_eeg[eeg_ch][0][0]

    print(f"  EEG: {eeg_ch}  |  ECG: {ecg_ch}")
    print(f"  Duration: {raw_eeg.times[-1]/3600:.2f} h  "
          f"({raw_eeg.times[-1]/60:.1f} min)")

    # ── R-peak detection ─────────────────────────────────────────────────────
    ecg_filt = mne.filter.filter_data(
        ecg, SFREQ, l_freq=5, h_freq=40, verbose=False)
    ecg_sq   = ecg_filt ** 2
    peaks, _ = find_peaks(
        ecg_sq,
        distance=int(0.4 * SFREQ),
        height=np.percentile(ecg_sq, 90))

    rr_intervals = np.diff(peaks) / SFREQ
    hr_mean      = 60 / np.mean(rr_intervals)
    print(f"  R-peaks: {len(peaks)}  |  Mean HR: {hr_mean:.1f} bpm")

    # ectopic rejection: RR > 2 SD from mean
    rr_mean    = np.mean(rr_intervals)
    rr_std     = np.std(rr_intervals)
    valid_peaks = [peaks[0]]
    for i, p in enumerate(peaks[1:], 1):
        rr = (p - peaks[i-1]) / SFREQ
        if abs(rr - rr_mean) < 2 * rr_std:
            valid_peaks.append(p)

    n_ectopic = len(peaks) - len(valid_peaks)
    print(f"  Ectopic removed: {n_ectopic}")
    peaks = np.array(valid_peaks)

    # ── HEP extraction ───────────────────────────────────────────────────────
    hep_s  = int(HEP_START_MS  / 1000 * SFREQ)
    hep_e  = int(HEP_END_MS    / 1000 * SFREQ)
    base_s = int(BASE_START_MS / 1000 * SFREQ)   # negative
    base_e = int(BASE_END_MS   / 1000 * SFREQ)   # 0

    hep_times  = []
    hep_amps   = []
    n_rejected = 0

    for p in peaks:
        b_start = p + base_s
        b_end   = p + base_e
        e_start = p + hep_s
        e_end   = p + hep_e

        if b_start < 0 or e_end >= len(eeg):
            n_rejected += 1
            continue

        baseline_data = eeg[b_start:b_end]
        epoch_data    = eeg[e_start:e_end]

        if (np.max(np.abs(baseline_data)) * 1e6 > REJECT_UV or
                np.max(np.abs(epoch_data))    * 1e6 > REJECT_UV):
            n_rejected += 1
            continue

        hep_times.append(p / SFREQ)
        hep_amps.append(np.mean(epoch_data - np.mean(baseline_data)) * 1e6)

    hep_times = np.array(hep_times)
    hep_amps  = np.array(hep_amps)
    print(f"  Valid epochs: {len(hep_times)}  |  "
          f"Rejected: {n_rejected} "
          f"({100*n_rejected/(len(hep_times)+n_rejected):.1f}%)")

    if len(hep_times) < WINDOW_MIN * 60:
        print(f"  [SKIP] {tag}: too few epochs for rolling variance")
        return None

    # ── rolling variance ─────────────────────────────────────────────────────
    beats_per_sec = len(hep_times) / (hep_times[-1] - hep_times[0])
    window_beats  = int(WINDOW_MIN * 60 * beats_per_sec)
    print(f"  Rolling window: {window_beats} beats (~{WINDOW_MIN} min)")

    rolling_var  = np.full(len(hep_amps), np.nan)
    rolling_mean = np.full(len(hep_amps), np.nan)
    for i in range(window_beats, len(hep_amps)):
        w = hep_amps[i - window_beats:i]
        rolling_var[i]  = np.var(w)
        rolling_mean[i] = np.mean(w)

    # ── DEWS summary ─────────────────────────────────────────────────────────
    print(f"\n  {'#':<4} {'Type':<32} {'Early':>8} {'Late':>8} "
          f"{'Ratio':>7} {'Dir':>6} {'n_e':>6} {'n_l':>6}")
    print(f"  {'-'*80}")

    results = []
    for si, (_, row) in enumerate(seizures.iterrows()):
        onset_s = row['onset']
        early_s = onset_s - EARLY_MIN[0] * 60
        early_e = onset_s - EARLY_MIN[1] * 60
        late_s  = onset_s - LATE_MIN[0]  * 60
        late_e  = onset_s - LATE_MIN[1]  * 60

        m_early = (hep_times >= early_s) & (hep_times < early_e)
        m_late  = (hep_times >= late_s)  & (hep_times < late_e)

        if m_early.sum() > 5 and m_late.sum() > 5:
            var_e = float(np.nanmean(rolling_var[m_early]))
            var_l = float(np.nanmean(rolling_var[m_late]))
            # skip if either window is still in the NaN warm-up region
            if np.isnan(var_e) or np.isnan(var_l) or var_e == 0:
                print(f"  {si+1:<4} {row['eventType']:<32} "
                      f"{'insufficient data (NaN)':>40}")
                continue
            ratio = var_l / var_e
            dirn  = ("UP ↑"   if ratio > 1.05 else
                     "DOWN ↓" if ratio < 0.95 else
                     "FLAT →")
            results.append(dict(
                sub=sub_id, run=run_id, seizure=si + 1,
                type=row['eventType'],
                var_early=var_e, var_late=var_l,
                ratio=ratio, direction=dirn,
                n_early=int(m_early.sum()), n_late=int(m_late.sum())))
            print(f"  {si+1:<4} {row['eventType']:<32} {var_e:>8.3f} "
                  f"{var_l:>8.3f} {ratio:>7.2f} {dirn:>6} "
                  f"{m_early.sum():>6} {m_late.sum():>6}")
        else:
            print(f"  {si+1:<4} {row['eventType']:<32} "
                  f"{'insufficient data':>40}")

    if not results:
        return None

    # ── plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(seizures), 2,
                             figsize=(16, 4.5 * len(seizures)))
    if len(seizures) == 1:
        axes = axes.reshape(1, -1)

    for si, (_, row) in enumerate(seizures.iterrows()):
        onset_s = row['onset']
        label   = row['eventType'].replace('sz_foc_', '')
        start_s = onset_s - PRE_ICTAL_MIN * 60

        mask = (hep_times >= start_s) & (hep_times <= onset_s + 120)
        t    = (hep_times[mask] - onset_s) / 60
        amp  = hep_amps[mask]
        var  = rolling_var[mask]
        mean = rolling_mean[mask]

        for ax_col, data, ylabel, color, title_sfx in [
            (0, amp, "HEP amplitude (µV)", TEAL,  "HEP mean"),
            (1, var, "Variance (µV²)",     AMBER, "HEP variance — DEWS"),
        ]:
            ax = axes[si, ax_col]
            if ax_col == 0:
                ax.plot(t, amp,  lw=0.5, color=LGREY, alpha=0.5)
                ax.plot(t, mean, lw=2.0, color=color, label='Rolling mean')
            else:
                ax.plot(t, data, lw=2.0, color=color, label='Rolling variance')

            ax.axvspan(-EARLY_MIN[0], -EARLY_MIN[1],
                       alpha=0.08, color='blue', label='Early window')
            ax.axvspan(-LATE_MIN[0], 0,
                       alpha=0.08, color='red',  label='Late window')
            ax.axvline(0, color=RED, lw=1.5, ls='--', label='Seizure onset')
            ax.set_title(f"Seizure {si+1} — {label} — {title_sfx}",
                         fontsize=10)
            ax.set_xlabel("Time to onset (min)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.suptitle(
        f"{sub_id} {run_id} — DEWS negative control (epileptic seizures)\n"
        "HRIT v3 prediction: no pre-ictal variance rise in epilepsy",
        fontsize=12, color=NAVY, y=1.01)
    plt.tight_layout()

    fig_path = base_path / f"dews_{sub_id}_{run_id}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {fig_path.name}")

    return pd.DataFrame(results)


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    base = Path(".")

    # subject → runs to analyse (runs confirmed to contain sz_foc events)
    subjects = {
        "sub-047": ["run-01", "run-02", "run-03", "run-04",
                    "run-05", "run-06", "run-07"],
        "sub-035": ["run-01", "run-02", "run-03", "run-04", "run-05",
                    "run-06", "run-07", "run-08", "run-09", "run-10",
                    "run-11", "run-12", "run-13", "run-14"],
    }

    # allow overriding subjects from command line, e.g. python dews_multi.py sub-001
    if len(sys.argv) > 1:
        override = sys.argv[1:]
        subjects = {s: subjects.get(s, ["run-01"]) for s in override
                    if s in subjects}
        if not subjects:
            # fallback: all runs for the given subject
            subjects = {sys.argv[1]: [
                f"run-{i:02d}" for i in range(1, 15)]}

    all_results = []
    for sub_id, runs in subjects.items():
        sub_results = []
        for run_id in runs:
            df = run_dews(sub_id, run_id, base_path=base)
            if df is not None:
                sub_results.append(df)

        if sub_results:
            sub_df = pd.concat(sub_results, ignore_index=True)
            csv_path = base / f"dews_{sub_id}_results.csv"
            sub_df.to_csv(csv_path, index=False)
            print(f"\nSaved: {csv_path.name}  ({len(sub_df)} seizures)")

            # per-subject summary
            print(f"\n── {sub_id} SUMMARY ──")
            for dirn, grp in sub_df.groupby('direction'):
                print(f"  {dirn}: {len(grp)} seizure(s)  "
                      f"(mean ratio {grp['ratio'].mean():.2f})")

            all_results.append(sub_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(base / "dews_all_results.csv", index=False)
        print(f"\nCombined CSV saved: dews_all_results.csv "
              f"({len(combined)} total seizures)")


if __name__ == "__main__":
    main()
