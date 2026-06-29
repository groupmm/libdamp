import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.fft import fft
from scipy.signal import butter, resample, sosfiltfilt

from libdamp.helpers.freq import midi2hz


class PulseItConverter:
    """Helper class to convert between the raw audio and the F0/FC/Gain tracks for the PULSE-IT method."""

    def __init__(self) -> None:
        super().__init__()

    def pulse_it_analysis(
        self,
        wav_file,
        target_fs=None,
        notes_csv_file=None,
        f0_csv_file=None,
        f0_col_names=["TIME", "VALUE"],
        pulse_marker_csv_file=None,
        separator=",",
        target_num_samples=None,
    ):

        # inizialize empty annotations
        notes_annotation = []
        f0_annotation = []
        pulse_annotation = []

        # load annotations, discern between different annotation formats
        if notes_csv_file:
            notes_annotation = pd.read_csv(notes_csv_file, sep=separator, skipinitialspace=True)

        if f0_csv_file:
            f0_annotation = pd.read_csv(f0_csv_file, skipinitialspace=True)
            f0_annotation = f0_annotation.rename(columns={f0_col_names[0]: "TIME", f0_col_names[1]: "VALUE"})
            # Filter out zero or negative f0 values
            f0_annotation = f0_annotation[f0_annotation["VALUE"] > 0].reset_index(drop=True)

        if pulse_marker_csv_file:
            pulse_annotation = pd.read_csv(pulse_marker_csv_file, sep=separator)

        # load and normalize
        x, fs = sf.read(wav_file, dtype=np.float32)
        x *= 1.0 / np.max(np.abs(x))
        num_samples = len(x)

        # preset some parameters
        if target_fs is None:
            target_fs = fs
        stft_blocksize = 2048
        num_partials = 10

        # resample if necessary
        if fs != target_fs:
            x = resample(x, int(num_samples * float(target_fs) / float(fs)))
            num_samples = len(x)
            fs = target_fs

        # zero-pad if necessary
        if target_num_samples:
            if num_samples < target_num_samples:
                x = np.pad(x, (0, target_num_samples - num_samples))
                x += 0.0000000001 * np.random.randn(target_num_samples)
                num_samples = target_num_samples

        # decide whether to use fundamental frequency annotation
        # or pulse marker annotation
        if f0_csv_file:
            num_rows = len(f0_annotation)
            pulse_samples = np.zeros(num_rows, dtype=np.int32)
            fund_period = 100 * np.ones(num_rows, dtype=np.int32)
            fund_freq = np.zeros(num_rows)

            for index, row in f0_annotation.iterrows():
                start_time = int(fs * float(row["TIME"]))
                pulse_samples[index] = start_time
                curr_f0 = float(row["VALUE"])

                win_len = int(fs / curr_f0)

                fund_freq[index] = curr_f0
                fund_period[index] = win_len
        elif pulse_marker_csv_file:
            num_rows = len(pulse_annotation)
            pulse_samples = np.zeros(num_rows, dtype=np.int32)
            fund_period = 100 * np.ones(num_rows, dtype=np.int32)
            fund_freq = np.zeros(num_rows)

            # get zero crossings
            zero_crossings = np.where((x[1:] >= 0.0) & (x[:-1] <= 0.0))[0]
            zero_crossings_in_seconds = zero_crossings / float(fs)
            pulse_samples = np.zeros(num_rows, dtype=np.int32)

            for index, row in pulse_annotation.iterrows():
                pulse_start = float(row["TIME"])
                difference = pulse_start - zero_crossings_in_seconds
                min_ind = np.argmin(np.abs(difference))
                pulse_samples[index] = zero_crossings[min_ind]

            fund_period[:-1] = pulse_samples[1:] - pulse_samples[:-1]
            fund_freq = fs / (np.maximum(1, fund_period))

        # now make pitch-synchronous STFT
        numPulses = len(pulse_samples)
        X = np.zeros((stft_blocksize, numPulses), dtype=np.complex64)

        for k in range(numPulses):
            # take a signal snippet of approximately the fundamental perid duration
            snip = x[pulse_samples[k] : (pulse_samples[k] + int(fund_period[k]))]

            # and reseample it to a standardized length
            snip = resample(snip, stft_blocksize)

            f = fft(snip, axis=0)

            # store into STFT matrix
            X[:, k] = f

        # compute spectral centroid according to eq (1) in Horner and Beauchamp 1995
        partial_magnitudes = np.abs(X[:num_partials, :])
        partial_weights = 0.0 + np.expand_dims(np.arange(num_partials), 0)

        SC = fund_freq * ((np.matmul(partial_weights, partial_magnitudes)) / (np.sum(partial_magnitudes, axis=0)) - 1.0)
        SC = np.squeeze(SC)

        # compute simple RMS
        RMS = np.sqrt(np.mean(np.abs(X) ** 2.0, axis=0))

        # prevent edge effects of the spectral centroid in case of f0 annotation
        if f0_csv_file:
            SC[np.where(RMS < 1)] = 0.0

        return num_samples, pulse_samples, fund_freq, SC, RMS, x, notes_annotation, target_fs

    def adjustControlSignals(
        self,
        num_samples,
        frame_segments,
        fund_freq,
        SC,
        RMS,
        transpose_f0_semitones=0.0,
        adjust_fc_semitones=0.0,
        smoothing_filter_cutoff=100.0,
        flatten_pitch=False,
        fs=None,
    ):
        # get number of elements
        numFrames = len(fund_freq)
        lowest_freq = midi2hz(torch.tensor(0.0))

        # fundamental frequency transposition
        f0_transpose_factor = np.power(2.0, transpose_f0_semitones / 12.0)

        # filter cutoff adjustment
        fc_transpose_factor = np.power(2.0, adjust_fc_semitones / 12.0)
        brightness_factor = 4.5
        brightness_offset = -200.0

        transposed_spec_cent = fc_transpose_factor * brightness_factor * SC + brightness_offset

        if flatten_pitch:
            constant_pitch = torch.tensor(60.0)  # a C3
            transposed_fund_freq = 0.0 * fund_freq + f0_transpose_factor * midi2hz(constant_pitch)
        else:
            transposed_fund_freq = f0_transpose_factor * fund_freq

        # now blow-up to full sampling-rate, hereby the frame segments determine the spacing between sampling points
        # they don't need to be equidistant, they can also represent note objects or fundamental period durations
        f0_track = np.zeros(num_samples)
        fc_track = np.zeros(num_samples)
        gain_track = np.zeros(num_samples)

        # fill in the main tones, these can either be symbolic note events
        # or audio frames, all we need is start and end times, as well as
        # F0, Fc, and Gain per tones / frames
        for k in range(numFrames - 1):
            start_sample = frame_segments[k]
            end_sample = frame_segments[k + 1]
            f0_track[start_sample:end_sample] = np.maximum(lowest_freq, transposed_fund_freq[k])
            fc_track[start_sample:end_sample] = np.maximum(lowest_freq, transposed_spec_cent[k])
            gain_track[start_sample:end_sample] = RMS[k] / np.max(RMS)

        # special treatment for the borders, to prevent smoothing issues
        # first the beginning
        start_sample = 0
        end_sample = frame_segments[0]
        f0_track[start_sample:end_sample] = np.maximum(lowest_freq, transposed_fund_freq[0])
        fc_track[start_sample:end_sample] = lowest_freq
        gain_track[start_sample:end_sample] = 0.0

        # second the ending
        start_sample = frame_segments[-1]
        end_sample = num_samples
        f0_track[start_sample:end_sample] = np.maximum(lowest_freq, transposed_fund_freq[-1])
        fc_track[start_sample:end_sample] = lowest_freq
        gain_track[start_sample:end_sample] = 0.0

        # smooth the resulting trajectories with a zero phase, 2nd order butterworth filter
        sos = butter(2, Wn=np.clip(a_max=0.4999 * fs, a_min=lowest_freq, a=smoothing_filter_cutoff), btype="low", fs=float(fs), output="sos")

        f0_track = sosfiltfilt(sos, f0_track)
        fc_track = sosfiltfilt(sos, fc_track)
        gain_track = sosfiltfilt(sos, gain_track)

        # prevent overshoot into very low range of values
        f0_track = np.maximum(lowest_freq, f0_track)
        fc_track = np.maximum(lowest_freq, fc_track)
        gain_track = np.maximum(0.0, gain_track)

        return f0_track, fc_track, gain_track

    def convert(
        self,
        wav_file,
        target_fs=None,
        notes_csv_file=None,
        f0_csv_file=None,
        f0_col_names=None,
        pulse_marker_csv_file=None,
        separator=",",
        target_num_samples=None,
    ):
        num_samples, pulse_samples, fund_freq, SC, RMS, x, notes_annotation, fs = self.pulse_it_analysis(
            wav_file=wav_file,
            target_fs=target_fs,
            notes_csv_file=notes_csv_file,
            f0_csv_file=f0_csv_file,
            f0_col_names=f0_col_names,
            pulse_marker_csv_file=pulse_marker_csv_file,
            separator=separator,
            target_num_samples=target_num_samples,
        )

        f0_track, fc_track, gain_track = self.adjustControlSignals(
            num_samples=num_samples,
            frame_segments=pulse_samples,
            fund_freq=fund_freq,
            SC=SC,
            RMS=RMS,
            transpose_f0_semitones=0.0,
            adjust_fc_semitones=0.0,
            smoothing_filter_cutoff=100.0,
            flatten_pitch=False,
            fs=fs,
        )

        return x, fs, f0_track, fc_track, gain_track


def main() -> None:
    """Command-line entry point to preprocess audio files for the PULSE-IT dataset."""
    parser = argparse.ArgumentParser(description="Preprocess audio files for the PULSE-IT dataset.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input .wav files")
    parser.add_argument("ann_dir", type=str, help="Directory containing the annotation .csv files")
    parser.add_argument("col_names", type=str, nargs=2, default=["t", "f0"], help="Column names in the annotation .csv files for time and f0 values")
    parser.add_argument("output_dir", type=str, default="processed_data", help="Directory to save the processed .npy files")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ann_dir = Path(args.ann_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = PulseItConverter()

    for wav_file in audio_dir.glob("**/*.wav"):
        # Look for the annotation file in ann_dir and its subfolders
        basename = wav_file.stem + ".csv"
        f0_csv_file = None
        for csv_file in ann_dir.glob("**/*.csv"):
            if csv_file.name == basename:
                f0_csv_file = csv_file
                break
        if f0_csv_file is None:
            print(f"Warning: No annotation file found for {wav_file}, skipping...")
            continue

        x, fs, f0_track, fc_track, gain_track = converter.convert(
            wav_file=wav_file,
            f0_csv_file=f0_csv_file,
            f0_col_names=args.col_names,
            target_fs=None,
        )

        # Save as .npy files for later loading in the dataset class
        sf.write(output_dir / f"{wav_file.stem}.wav", x, fs)
        np.save(output_dir / f"{wav_file.stem}_f0.npy", f0_track)
        np.save(output_dir / f"{wav_file.stem}_fc.npy", fc_track)
        np.save(output_dir / f"{wav_file.stem}_gain.npy", gain_track)


if __name__ == "__main__":
    main()
