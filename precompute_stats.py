"""
Precompute the statistics the game needs and save them as CSV files.

Run this once (or whenever the dataset changes):
    python precompute_stats.py

Outputs (written next to this script):
    overall_mode.csv       position, mode
    class_value_freq.csv   class, position, value, frequency
                           (frequency = fraction of that class's sequences
                           with this value at this position; zeros omitted)
"""

import csv
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PKL_PATH = HERE / "nuclease_design_processed.pkl"
MODE_CSV = HERE / "overall_mode.csv"
FREQ_CSV = HERE / "class_value_freq.csv"

CLASSES = [0, 1, 2, 3]


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    X_train = np.asarray(data["X_train_int"])
    y_train = np.asarray(data["y_train"]).flatten()
    print("Training data loaded:", X_train.shape)

    sequence_length = X_train.shape[1]
    vocab_size = int(X_train.max()) + 1

    # ---- overall mode at each position ----
    overall_mode = np.array([
        np.bincount(X_train[:, j], minlength=vocab_size).argmax()
        for j in range(sequence_length)
    ])

    with open(MODE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "mode"])
        for j, m in enumerate(overall_mode):
            writer.writerow([j, int(m)])

    # ---- per-class value frequencies ----
    n_rows = 0
    with open(FREQ_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "position", "value", "frequency"])

        for c in CLASSES:
            class_data = X_train[y_train == c]
            class_size = len(class_data)
            if class_size == 0:
                print(f"Warning: class {c} has no sequences")
                continue

            for j in range(sequence_length):
                counts = np.bincount(class_data[:, j], minlength=vocab_size)
                for v in np.nonzero(counts)[0]:
                    writer.writerow([c, j, int(v), float(counts[v] / class_size)])
                    n_rows += 1

    print(f"Wrote {MODE_CSV.name} ({sequence_length} rows)")
    print(f"Wrote {FREQ_CSV.name} ({n_rows} rows)")


if __name__ == "__main__":
    main()
