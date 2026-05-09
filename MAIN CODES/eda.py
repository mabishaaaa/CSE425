# -*- coding: utf-8 -*-
"""
eda.py

Exploratory Data Analysis for:
Unsupervised Neural Network for Multi-Genre Music Generation

This script analyzes the MAESTRO MIDI dataset and produces the EDA outputs
required by the supplementary implementation guide.

It generates:
1. Duration histogram
2. Split distribution plot
3. Note count distribution
4. Pitch distribution histogram
5. Velocity distribution histogram
6. Piano-roll sparsity histogram
7. Active-cell ratio histogram
8. Window count summary
9. Example piano-roll image
10. CSV summaries

Default paths follow the former Colab-style project structure.
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

# DEFAULT_BASE = "/content/drive/MyDrive/music-project"
# DEFAULT_CSV_PATH = "/content/maestro/maestro-v3.0.0/maestro-v3.0.0.csv"
# DEFAULT_MIDI_ROOT = "/content/maestro/maestro-v3.0.0"

DEFAULT_BASE = "."
DEFAULT_CSV_PATH = "/Users/maliha/Downloads/CSE425 Project copy/drive-download-20260508T224556Z-3-001/Music Project/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.csv"
DEFAULT_MIDI_ROOT = "maestro-v3.0.0"

PIANO_MIN = 21
PIANO_MAX = 108
PIANO_KEYS = 88

FS = 16
SEQ_LEN = 128
SPARSITY_THRESHOLD = 0.02


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def load_midi_safe(path: str):
    try:
        return pretty_midi.PrettyMIDI(path)
    except Exception as e:
        return None


def extract_notes(pm: pretty_midi.PrettyMIDI):
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    return notes


def compute_piano_roll_stats(pm: pretty_midi.PrettyMIDI):
    """
    Returns:
        active_ratio: fraction of active cells in piano-roll
        sparsity: fraction of zero cells
        total_windows: number of 128-step windows
        kept_windows: windows with active_ratio >= 0.02
        roll_shape: shape of piano-roll after slicing/transposing
        sample_roll: first valid roll segment for visualization
    """

    roll = pm.get_piano_roll(fs=FS)

    # Keep piano range A0-C8, shape becomes (88, T)
    roll = roll[PIANO_MIN:PIANO_MAX + 1, :]

    # Transpose to (T, 88)
    roll = roll.T

    # Binarize
    roll = (roll > 0).astype(np.float32)

    if roll.size == 0:
        return {
            "active_ratio": 0.0,
            "sparsity": 1.0,
            "total_windows": 0,
            "kept_windows": 0,
            "roll_shape": str(roll.shape),
            "sample_roll": None,
        }

    active_ratio = float(np.mean(roll))
    sparsity = float(1.0 - active_ratio)

    total_windows = 0
    kept_windows = 0

    for start in range(0, len(roll) - SEQ_LEN, SEQ_LEN):
        window = roll[start:start + SEQ_LEN]
        total_windows += 1

        if np.mean(window) >= SPARSITY_THRESHOLD:
            kept_windows += 1

    sample_roll = None
    if len(roll) >= SEQ_LEN:
        sample_roll = roll[:SEQ_LEN]

    return {
        "active_ratio": active_ratio,
        "sparsity": sparsity,
        "total_windows": total_windows,
        "kept_windows": kept_windows,
        "roll_shape": str(roll.shape),
        "sample_roll": sample_roll,
    }


def save_hist(values, title, xlabel, ylabel, save_path, bins=50):
    values = [v for v in values if pd.notna(v)]

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")


def save_bar(x, y, title, xlabel, ylabel, save_path, rotation=0):
    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")


def save_split_duration_boxplot(df, save_path):
    splits = ["train", "validation", "test"]
    data = []

    for s in splits:
        vals = df[df["split"] == s]["duration"].dropna().values
        if len(vals) > 0:
            data.append(vals)

    existing_splits = [
        s for s in splits if len(df[df["split"] == s]["duration"].dropna()) > 0
    ]

    if not data:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=existing_splits)
    plt.title("Duration Distribution by Split")
    plt.xlabel("Dataset Split")
    plt.ylabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")


def save_piano_roll_image(sample_roll, save_path):
    if sample_roll is None:
        print("No piano-roll sample available for visualization.")
        return

    plt.figure(figsize=(12, 5))
    plt.imshow(
        sample_roll.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    plt.title("Example Binary Piano-Roll Window")
    plt.xlabel("Time Step")
    plt.ylabel("Piano Key Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# MAIN EDA LOGIC
# ============================================================

def run_eda(
    csv_path: str,
    midi_root: str,
    base_dir: str,
    max_files: int | None = None,
    sample_for_roll_image: bool = True
):
    warnings.filterwarnings("ignore")

    plot_dir = os.path.join(base_dir, "outputs", "plots", "eda")
    output_dir = os.path.join(base_dir, "outputs")

    ensure_dir(plot_dir)
    ensure_dir(output_dir)

    print("=" * 70)
    print("MAESTRO EDA")
    print("=" * 70)
    print(f"CSV path     : {csv_path}")
    print(f"MIDI root    : {midi_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Plot dir     : {plot_dir}")
    print(f"FS           : {FS}")
    print(f"SEQ_LEN      : {SEQ_LEN}")
    print(f"Piano range  : {PIANO_MIN}-{PIANO_MAX}")
    print(f"Sparse filter: active ratio >= {SPARSITY_THRESHOLD}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["split", "duration", "midi_filename"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    if max_files is not None and max_files > 0:
        df_process = df.head(max_files).copy()
        print(f"Processing first {max_files} files only.")
    else:
        df_process = df.copy()
        print(f"Processing all {len(df_process)} files.")

    # ========================================================
    # BASIC CSV-LEVEL EDA
    # ========================================================

    print("\nDataset columns:")
    print(list(df.columns))

    print("\nSplit counts:")
    print(df["split"].value_counts())

    # Save split counts
    split_counts = df["split"].value_counts().reset_index()
    split_counts.columns = ["split", "count"]
    split_counts.to_csv(
        os.path.join(output_dir, "eda_split_counts.csv"),
        index=False
    )

    # Duration histogram
    save_hist(
        df["duration"].dropna().values,
        title="MAESTRO Duration Distribution",
        xlabel="Duration (seconds)",
        ylabel="Number of Recordings",
        save_path=os.path.join(plot_dir, "duration_histogram.png"),
        bins=50
    )

    # Duration by split
    save_split_duration_boxplot(
        df,
        save_path=os.path.join(plot_dir, "duration_by_split_boxplot.png")
    )

    # Split bar plot
    save_bar(
        split_counts["split"],
        split_counts["count"],
        title="MAESTRO Split Distribution",
        xlabel="Split",
        ylabel="Number of Recordings",
        save_path=os.path.join(plot_dir, "split_distribution.png")
    )

    # Composer distribution if available
    if "canonical_composer" in df.columns:
        top_composers = (
            df["canonical_composer"]
            .fillna("Unknown")
            .value_counts()
            .head(15)
            .reset_index()
        )
        top_composers.columns = ["composer", "count"]

        top_composers.to_csv(
            os.path.join(output_dir, "eda_top_composers.csv"),
            index=False
        )

        save_bar(
            top_composers["composer"],
            top_composers["count"],
            title="Top 15 Composers in MAESTRO",
            xlabel="Composer",
            ylabel="Number of Recordings",
            save_path=os.path.join(plot_dir, "top_composers.png"),
            rotation=45
        )

    # ========================================================
    # MIDI-LEVEL EDA
    # ========================================================

    pitch_counts = np.zeros(PIANO_KEYS, dtype=np.int64)
    velocity_values = []
    note_counts = []
    active_ratios = []
    sparsities = []
    total_windows_list = []
    kept_windows_list = []
    skipped_files = []

    file_rows = []
    example_roll = None

    print("\nProcessing MIDI files...")

    for idx, row in df_process.iterrows():
        midi_rel = row["midi_filename"]
        midi_path = os.path.join(midi_root, midi_rel)

        if not os.path.exists(midi_path):
            skipped_files.append({
                "midi_filename": midi_rel,
                "reason": "file_not_found"
            })
            continue

        pm = load_midi_safe(midi_path)

        if pm is None:
            skipped_files.append({
                "midi_filename": midi_rel,
                "reason": "parse_error"
            })
            continue

        notes = extract_notes(pm)

        piano_notes = [
            n for n in notes
            if PIANO_MIN <= n.pitch <= PIANO_MAX
        ]

        # Note count
        note_count = len(piano_notes)
        note_counts.append(note_count)

        # Pitch and velocity
        for note in piano_notes:
            pitch_counts[note.pitch - PIANO_MIN] += 1
            velocity_values.append(note.velocity)

        # Piano-roll stats
        roll_stats = compute_piano_roll_stats(pm)

        active_ratios.append(roll_stats["active_ratio"])
        sparsities.append(roll_stats["sparsity"])
        total_windows_list.append(roll_stats["total_windows"])
        kept_windows_list.append(roll_stats["kept_windows"])

        if sample_for_roll_image and example_roll is None:
            if roll_stats["sample_roll"] is not None:
                example_roll = roll_stats["sample_roll"]

        file_rows.append({
            "midi_filename": midi_rel,
            "split": row.get("split", "unknown"),
            "composer": row.get("canonical_composer", "unknown"),
            "title": row.get("canonical_title", "unknown"),
            "duration_csv": row.get("duration", np.nan),
            "duration_midi": pm.get_end_time(),
            "note_count": note_count,
            "active_ratio": roll_stats["active_ratio"],
            "sparsity": roll_stats["sparsity"],
            "total_windows": roll_stats["total_windows"],
            "kept_windows": roll_stats["kept_windows"],
            "roll_shape": roll_stats["roll_shape"],
        })

        if (len(file_rows) % 50) == 0:
            print(f"Processed {len(file_rows)} valid MIDI files...")

    file_stats = pd.DataFrame(file_rows)

    file_stats_path = os.path.join(output_dir, "eda_file_stats.csv")
    file_stats.to_csv(file_stats_path, index=False)
    print(f"\nSaved: {file_stats_path}")

    skipped_path = os.path.join(output_dir, "eda_skipped_files.csv")
    pd.DataFrame(skipped_files).to_csv(skipped_path, index=False)
    print(f"Saved: {skipped_path}")

    # ========================================================
    # MIDI-LEVEL PLOTS
    # ========================================================

    save_hist(
        note_counts,
        title="Note Count per MIDI Recording",
        xlabel="Number of Notes",
        ylabel="Number of Files",
        save_path=os.path.join(plot_dir, "note_count_distribution.png"),
        bins=50
    )

    save_bar(
        list(range(PIANO_MIN, PIANO_MAX + 1)),
        pitch_counts,
        title="Pitch Distribution Across Piano Range",
        xlabel="MIDI Pitch",
        ylabel="Note Count",
        save_path=os.path.join(plot_dir, "pitch_distribution.png")
    )

    save_hist(
        velocity_values,
        title="Velocity Distribution",
        xlabel="MIDI Velocity",
        ylabel="Number of Notes",
        save_path=os.path.join(plot_dir, "velocity_distribution.png"),
        bins=50
    )

    save_hist(
        sparsities,
        title="Piano-Roll Sparsity Distribution",
        xlabel="Sparsity Ratio",
        ylabel="Number of Files",
        save_path=os.path.join(plot_dir, "sparsity_distribution.png"),
        bins=40
    )

    save_hist(
        active_ratios,
        title="Piano-Roll Active Cell Ratio",
        xlabel="Active Cell Ratio",
        ylabel="Number of Files",
        save_path=os.path.join(plot_dir, "active_ratio_distribution.png"),
        bins=40
    )

    save_hist(
        kept_windows_list,
        title="Retained Training Windows per File",
        xlabel="Number of Retained Windows",
        ylabel="Number of Files",
        save_path=os.path.join(plot_dir, "retained_windows_distribution.png"),
        bins=50
    )

    save_piano_roll_image(
        example_roll,
        save_path=os.path.join(plot_dir, "example_piano_roll_window.png")
    )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary = {
        "total_csv_rows": len(df),
        "processed_rows": len(df_process),
        "valid_midi_files": len(file_stats),
        "skipped_files": len(skipped_files),
        "total_notes_processed": int(np.sum(note_counts)) if note_counts else 0,
        "mean_duration_csv": float(df["duration"].mean()),
        "median_duration_csv": float(df["duration"].median()),
        "min_duration_csv": float(df["duration"].min()),
        "max_duration_csv": float(df["duration"].max()),
        "mean_note_count": float(np.mean(note_counts)) if note_counts else 0.0,
        "median_note_count": float(np.median(note_counts)) if note_counts else 0.0,
        "mean_active_ratio": float(np.mean(active_ratios)) if active_ratios else 0.0,
        "mean_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "mean_total_windows_per_file": float(np.mean(total_windows_list)) if total_windows_list else 0.0,
        "mean_kept_windows_per_file": float(np.mean(kept_windows_list)) if kept_windows_list else 0.0,
        "total_windows": int(np.sum(total_windows_list)) if total_windows_list else 0,
        "total_kept_windows": int(np.sum(kept_windows_list)) if kept_windows_list else 0,
        "fs": FS,
        "seq_len": SEQ_LEN,
        "piano_min": PIANO_MIN,
        "piano_max": PIANO_MAX,
        "sparsity_threshold_active_ratio": SPARSITY_THRESHOLD,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "eda_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved: {summary_path}")

    print("\n" + "=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)

    for k, v in summary.items():
        print(f"{k:35s}: {v}")

    print("=" * 70)

    print("\nEDA complete.")
    print(f"Plots saved in: {plot_dir}")
    print(f"CSV summaries saved in: {output_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full EDA for MAESTRO music generation project."
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to maestro-v3.0.0.csv"
    )

    parser.add_argument(
        "--midi-root",
        type=str,
        default=DEFAULT_MIDI_ROOT,
        help="Root directory containing MAESTRO MIDI files"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE,
        help="Base project directory. Outputs are saved under base-dir/outputs/"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of MIDI files for faster EDA. Use 0 for all files."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    max_files = None if args.max_files == 0 else args.max_files

    run_eda(
        csv_path=args.csv_path,
        midi_root=args.midi_root,
        base_dir=args.base_dir,
        max_files=max_files,
        sample_for_roll_image=True
    )