import mne
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. FIND ONE SUBJECT ──────────────────────────────────────────────────────
repo = Path(".")  # adjust if needed
edfs = sorted(repo.rglob("*.edf"))
print(f"Total EDF files found: {len(edfs)}")
print("First 5:", [str(e) for e in edfs[:5]])

# ── 2. LOAD FIRST EDF ────────────────────────────────────────────────────────
if not edfs:
    print("No EDF files found.")
    exit(1)

edf_path = edfs[0]
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

print(f"\nFile: {edf_path.name}")
print(f"Channels ({len(raw.ch_names)}): {raw.ch_names}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]/60:.1f} minutes")

# ── 3. FIND ECG CHANNEL ──────────────────────────────────────────────────────
ecg_candidates = [c for c in raw.ch_names 
                  if any(k in c.upper() for k in ['ECG','EKG','HEART','CARD'])]
print(f"\nECG candidates: {ecg_candidates}")

# ── 4. FIND ANNOTATIONS FILE ─────────────────────────────────────────────────
# BIDS: same folder, _events.tsv or _annotations.tsv
tsv_files = sorted(edf_path.parent.glob("*.tsv"))
print(f"\nTSV files in same folder: {[t.name for t in tsv_files]}")

if tsv_files:
    annot = pd.read_csv(tsv_files[0], sep='\t')
    print(f"\nAnnotation columns: {list(annot.columns)}")
    print(annot.head(10))
    
    # Find seizure events
    seiz_mask = annot.apply(lambda r: r.astype(str).str.contains(
        'seiz|ictal|onset', case=False).any(), axis=1)
    seizures = annot[seiz_mask]
    print(f"\nSeizure rows found: {len(seizures)}")
    if len(seizures):
        print(seizures.head())

# ── 5. QUICK ECG PLOT ─────────────────────────────────────────────────────────
if ecg_candidates:
    ecg_ch = ecg_candidates[0]
    ecg_data, times = raw[ecg_ch, :int(raw.info['sfreq']*60)]  # first 60s
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, ecg_data[0]*1e6, lw=0.8, color='#028090')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"ECG — {edf_path.name} — first 60 seconds")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("smoke_test_ecg.png", dpi=150)
    print("\nECG plot saved: smoke_test_ecg.png")
else:
    print("\nNo ECG channel found — check channel names above")

# ── 6. SUMMARY ────────────────────────────────────────────────────────────────
print("\n── SMOKE TEST SUMMARY ──")
print(f"EDF loads:        YES")
print(f"Channel count:    {len(raw.ch_names)}")
print(f"Sfreq:            {raw.info['sfreq']} Hz")
print(f"ECG present:      {'YES — ' + ecg_candidates[0] if ecg_candidates else 'NOT FOUND'}")
print(f"Annotations:      {'YES' if tsv_files else 'NOT FOUND'}")
