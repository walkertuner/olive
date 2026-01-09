#!/usr/bin/env python3

import os
import glob
import argparse
import math
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# -----------------------------
# TensorBoard utilities
# -----------------------------

def load_events(run_dir, tag):
    ea = EventAccumulator(
        run_dir,
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        return []

    return [(e.step, e.value) for e in ea.Scalars(tag)]


def best_step(run_dir, tag, mode="min"):
    events = load_events(run_dir, tag)
    if not events:
        raise RuntimeError(f"No events for tag '{tag}' in {run_dir}")

    steps, values = zip(*events)

    if mode == "min":
        idx = min(range(len(values)), key=lambda i: values[i])
    elif mode == "max":
        idx = max(range(len(values)), key=lambda i: values[i])
    else:
        raise ValueError("mode must be 'min' or 'max'")

    return steps[idx]


def value_at_step(run_dir, tag, step):
    events = dict(load_events(run_dir, tag))
    if step not in events:
        raise RuntimeError(
            f"Tag '{tag}' has no value at step {step} in {run_dir}"
        )
    return events[step]


# -----------------------------
# Statistics
# -----------------------------

def summarize(values, label):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1 if n > 1 else 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)

    print(label)
    print(f"  mean = {mean:.6f}")
    print(f"  std  = {std:.6f}")
    print(f"  sem  = {sem:.6f}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="log", help="Root TensorBoard log directory")
    p.add_argument("run", help="Run name (contains fold_*/ subdirs)")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    run_base = os.path.join(args.logdir, args.run)
    run_dirs = sorted(glob.glob(os.path.join(run_base, "fold_*")))

    if not run_dirs:
        raise RuntimeError(
            f"No folds found in {run_base} (expected fold_*/)"
        )

    print(f"Found {len(run_dirs)} folds\n")

    mae_vals = []
    octave_vals = []
    pitch_vals = []

    for run in run_dirs:
        fold = os.path.basename(run)

        # --- select epoch by MAE ---
        step = best_step(
            run,
            tag="mae/partials_total",
            mode="min",
        )

        # --- read all metrics at that same epoch ---
        mae = value_at_step(run, "mae/partials_total", step)
        octave = value_at_step(run, "accuracy/octave", step)
        pitch = value_at_step(run, "accuracy/pitch", step)

        mae_vals.append(mae)
        octave_vals.append(octave)
        pitch_vals.append(pitch)

        print(
            f"{fold:<20} "
            f"step={step:>6}  "
            f"MAE={mae:.4f}  "
            f"oct={octave:.4f}  "
            f"pitch={pitch:.4f}"
        )

    print("\n--- Cross-validation summary ---\n")

    summarize(mae_vals, "MAE (reference metric)")
    print()
    summarize(octave_vals, "Octave accuracy")
    print()
    summarize(pitch_vals, "Pitch accuracy")


if __name__ == "__main__":
    main()