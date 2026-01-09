import os
import glob
import argparse
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_scalar_series(run_dir, tag):
    """
    Load a scalar time series from a TensorBoard run directory.
    Returns a list of (step, value).
    """
    ea = event_accumulator.EventAccumulator(
        run_dir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(f"Tag '{tag}' not found in {run_dir}")

    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in events]


def best_value(run_dir, tag, mode="min"):
    """
    Return best scalar value in a run according to mode.
    mode: 'min' or 'max'
    """
    series = load_scalar_series(run_dir, tag)
    values = [v for _, v in series]

    if mode == "min":
        return min(values)
    elif mode == "max":
        return max(values)
    else:
        raise ValueError("mode must be 'min' or 'max'")


def summarize(values, label):
    values = np.asarray(values)
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    sem = std / np.sqrt(len(values)) if len(values) > 1 else 0.0

    print(f"{label}:")
    print(f"  mean = {mean:.4f}")
    print(f"  std  = {std:.4f}")
    print(f"  sem  = {sem:.4f}")
    print()

    return mean, std, sem


def main():
    args = parse_args()

    run_base = os.path.join(args.logdir, args.run)
    run_dirs = sorted(glob.glob(os.path.join(run_base, 'fold_*')))

    if not run_dirs:
        raise RuntimeError(f"No runs found in {args.logdir} matching {args.run}/fold_*")

    print(f"Found {len(run_dirs)} folds\n")

    # ---- Overall MAE per fold ----
    fold_maes = []
    for run in run_dirs:
        mae = best_value(run, args.tag, mode="min")
        fold_maes.append(mae)
        print(f"{os.path.basename(run):<20} best {args.tag}: {mae:.4f}")

    print()
    summarize(fold_maes, "Cross-validation MAE")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize cross-validation metrics from TensorBoard logs"
    )

    parser.add_argument("run", help="Name of run")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--tag",default="mae/partials_total")

    return parser.parse_args()
