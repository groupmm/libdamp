#!/usr/bin/env python

"""Sanity-check visualization for the LM-SVR baseline (experiments/TISMIR2024).

Loads the best checkpoint of a trained experiment (identified by its experiment ID, e.g.
"AAT3_LMA_1M_DT", matching a `configs/TISMIR2024/data_splits/*.gin` file), runs it on an
excerpt of that experiment's held-out test take, and plots the signal at each stage of the
signal flow (LM input, TI filter, peak norm, noise synth, TV filter) next to the CM
reference - similar in spirit to Figure 5 of the paper, as a quick visual check that the
reimplementation behaves plausibly (not a quantitative evaluation).

Usage
-----
    uv run experiments/TISMIR2024/visualize_baseline.py AAT3_LMA_1M_DT
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import gin
import matplotlib.pyplot as plt
import torch

# make experiments and scripts directly discoverable, as in `scripts/run.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.TISMIR2024.lmsvr_baseline  # noqa: F401 - registers LMSVRBaselineExperiment
import libdamp  # noqa: F401 - registers gin-configurable classes
import scripts.run  # noqa: F401 - registers the "libdamp" gin-configurable entry point

STAGES = ("x_lm", "ti_filtered", "peak_normed", "noisy", "y_hat", "x_cm")
TITLES = {
    "x_lm": "$x_{LM}$ (input)",
    "ti_filtered": "after TI filter",
    "peak_normed": "after peak norm",
    "noisy": "after noise synth",
    "y_hat": r"$\hat{x}_{CM}$ (output)",
    "x_cm": "$x_{CM}$ (reference)",
}


def find_run_dir(results_path: str, experiment_id: str) -> str:
    """Find the (most recent) results folder for an experiment ID, e.g. 'AAT3_LMA_1M_DT'."""
    matches = sorted(glob.glob(os.path.join(results_path, f"{experiment_id}_*")))
    if not matches:
        raise FileNotFoundError(f"No results folder matching '{experiment_id}_*' found in '{results_path}'.")
    if len(matches) > 1:
        print(f"Multiple runs found for '{experiment_id}', using the most recently modified one: {matches[-1]}")
    return max(matches, key=os.path.getmtime)


def find_best_checkpoint(run_dir: str) -> str:
    """Find the best checkpoint saved for a run (there is only one, since `save_best_n_ckpts=1`)."""
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_best-*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No best checkpoint found in '{run_dir}/checkpoints'.")
    return ckpts[-1]


def load_experiment_and_val_dataset(run_dir: str, device: str):
    """Reconstruct the experiment and its validation dataset from a run's saved operative config.

    This re-instantiates the experiment (with random weights, to be overwritten by a checkpoint)
    and the exact train/val/test datasets used for that run, from `run_dir/operative_config.gin` -
    i.e. the same mechanism `scripts/run.py` itself uses to go from a gin config to live objects.
    """
    with gin.unlock_config():
        gin.parse_config_file(os.path.join(run_dir, "operative_config.gin"), skip_unknown=False)
    gin.finalize()

    experiment, _, val_dataset, _ = scripts.run.libdamp_entry_point()
    if val_dataset is None:
        raise ValueError(f"Run '{run_dir}' was configured without a validation dataset.")

    experiment.eval()
    experiment.to(device)
    return experiment, val_dataset


def pick_loudest_excerpt(signal: torch.Tensor, fs: float, duration: float) -> tuple[int, int]:
    """Return the start/end sample indices of the `duration`-second window with the highest RMS energy.

    Used to pick a representative (likely singing, not silent) excerpt from the held-out take.
    """
    win = int(fs)
    excerpt_len = int(fs * duration)
    if signal.shape[-1] <= excerpt_len:
        return 0, signal.shape[-1]

    energies = [signal[i : i + win].pow(2).mean() for i in range(0, signal.shape[-1] - win, win)]
    center = int(torch.argmax(torch.stack(energies))) * win + win // 2

    start = min(max(0, center - excerpt_len // 2), signal.shape[-1] - excerpt_len)
    return start, start + excerpt_len


def spectrogram_db(x: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """Compute a log-magnitude (dB) spectrogram of a 1D signal."""
    window = torch.hann_window(n_fft).to(x)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
    return 20 * torch.log10(torch.abs(X) + 1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_id", type=str, help="experiment ID, e.g. 'AAT3_LMA_1M_DT' (a configs/TISMIR2024/data_splits/*.gin file)")
    parser.add_argument("--results-path", type=str, default="results", help="base folder results were saved in (default: 'results')")
    parser.add_argument("--duration", type=float, default=6.0, help="length in seconds of the excerpt to visualize (default: 6.0)")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to run inference on (default: 'cpu')")
    parser.add_argument("--top-db", type=float, default=80.0, help="dynamic range of the spectrogram plots in dB (default: 80.0)")
    parser.add_argument("--output", type=str, default=None, help="where to save the figure (default: '<run_dir>/<experiment_id>_fig5.png')")
    args = parser.parse_args()

    run_dir = find_run_dir(args.results_path, args.experiment_id)
    ckpt_path = find_best_checkpoint(run_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    experiment, val_dataset = load_experiment_and_val_dataset(run_dir, args.device)
    experiment.load_state_dict(torch.load(ckpt_path, map_location=args.device)["state_dict"])

    if val_dataset.num_takes != 1:
        print(f"Warning: expected exactly one held-out take in the validation set, found {val_dataset.num_takes}; using the first one.")

    fs = val_dataset.target_fs
    lm_full, cm_full = val_dataset.lm_signals[0], val_dataset.cm_signals[0]

    start, end = pick_loudest_excerpt(cm_full, fs, args.duration)
    x_lm = lm_full[start:end].unsqueeze(0).to(args.device)
    x_cm = cm_full[start:end].unsqueeze(0).to(args.device)

    with torch.no_grad():
        _, stages = experiment(x_lm, return_intermediates=True)
    stages["x_cm"] = x_cm

    specs = {key: spectrogram_db(stages[key].squeeze(0).cpu(), n_fft=1024, hop=256) for key in STAGES}
    vmax = max(spec.max().item() for spec in specs.values())
    vmin = vmax - args.top_db

    fig, axes = plt.subplots(1, len(STAGES), figsize=(3.2 * len(STAGES), 3.5), sharey=True)
    for ax, key in zip(axes, STAGES, strict=True):
        ax.imshow(specs[key], origin="lower", aspect="auto", extent=[0, args.duration, 0, fs / 2], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(TITLES[key])
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")

    fig.suptitle(f"{args.experiment_id}: baseline signal flow (cf. Fig. 5 of the paper)")
    fig.tight_layout()

    output = args.output or os.path.join(run_dir, f"{args.experiment_id}_fig5.png")
    fig.savefig(output, dpi=150)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
