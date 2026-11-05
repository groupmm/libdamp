"""Baseline system for larynx microphone singing voice reconstruction (LM-SVR).

Reimplements the baseline described in Section 5 of Schwär et al., "A Dataset of Larynx
Microphone Recordings for Singing Voice Reconstruction," TISMIR 7(1), 2024
(https://doi.org/10.5334/tismir.166).

A control network estimates parameters for a chain of differentiable DSP blocks that
transform the larynx microphone (LM) signal towards the close-up microphone (CM) signal: a
time-invariant FIR filter (EQ, constant over the duration of the input but still estimated
from it), peak normalization, a noise synthesizer (for fricatives), and a time-varying FIR
filter (vocal tract / formants) - matching the signal flow in Figure 5 of the paper.
"""

import math

import gin
import torch
import torchaudio as taudio

import libdamp


def peak_norm(x, eps=1e-7):
    """Normalize each signal in the batch to unit peak amplitude.
    """
    peak = x.abs().amax(dim=-1, keepdim=True)
    return x / (peak + eps)


@gin.configurable
class LMSVRBaselineExperiment(libdamp.Experiment):
    """Baseline system for larynx microphone singing voice reconstruction (LM-SVR)."""

    def __init__(
        self,
        fs=16000.0,
        frame_len=320,
        filt_len=320,
        n_fft_mel=2048,
        n_mels=64,
        f_min_mel=50.0,
        f_max_mel=8000.0,
        num_bands_ti=64,
        num_bands_tv=64,
        num_bands_noise=10,
        model_size_control=128,
        learning_rate=1e-3,
        loss_fn=None,
        **kwargs,
    ):
        """Baseline system for larynx microphone singing voice reconstruction (LM-SVR).

        Parameters
        ----------
        fs : float
            Sampling rate in Hz (default: 16000.0).
        frame_len : int
            Frame length in samples, both for the mel input to the control network and
            for the time-varying DDSP blocks (default: 320).
        filt_len : int
            FIR filter length in samples, shared by the TI filter, TV filter, and noise
            synthesizer (default: 320).
        n_fft_mel : int
            FFT/window size of the Hann-windowed STFT used for the mel input to the control
            network (default: 2048).
        n_mels : int
            Number of mel bands for the control network's input (default: 64).
        f_min_mel : float
            Lowest frequency of the mel input (default: 50.0).
        f_max_mel : float or None
            Highest frequency of the mel input (default: 8000.0).
        num_bands_ti : int
            Number of linearly spaced magnitude bands for the time-invariant FIR filter (default: 64).
            This filter is constant over the duration of the input signal, but its magnitudes are still estimated
            from that input (by averaging the control network's per-frame features over time).
        num_bands_tv : int
            Number of linearly spaced magnitude bands for the time-varying FIR filter (default: 64).
        num_bands_noise : int
            Number of magnitude bands for the noise synthesizer's filter (default: 10).
        model_size_control : int
            Feature dimensionality of the CNN output and recurrent hidden state.
        learning_rate : float
        loss_fn : torch.nn.Module or None
            Loss function comparing predicted and reference CM signal.
        """
        super().__init__(**kwargs)

        self.fs = fs
        self.frame_len = frame_len
        self.filt_len = filt_len

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn if loss_fn is not None else libdamp.MSSLoss()

        f_max_mel = f_max_mel if f_max_mel is not None else fs / 2
        self.mel_transform = taudio.transforms.MelSpectrogram(
            sample_rate=fs,
            n_fft=n_fft_mel,
            hop_length=frame_len,
            f_min=f_min_mel,
            f_max=f_max_mel,
            n_mels=n_mels,
            center=False,
            pad=(n_fft_mel - frame_len) // 2,
        )

        self.control_cnn = libdamp.ConvStack(n_mels, model_size_control)
        self.control_rnn = libdamp.BiLSTM(model_size_control, model_size_control // 2)

        self.ti_head = torch.nn.Linear(model_size_control, num_bands_ti)
        self.noise_head = torch.nn.Linear(model_size_control, num_bands_noise)
        self.tv_head = torch.nn.Linear(model_size_control, num_bands_tv)

        self.ti_filter = libdamp.processors.TVMagnitudesFilter(
            frame_len, filt_len, fs, center_freqs=torch.linspace(0, fs / 2, num_bands_ti), filter_phase="linear"
        )
        self.tv_filter = libdamp.processors.TVMagnitudesFilter(
            frame_len, filt_len, fs, center_freqs=torch.linspace(0, fs / 2, num_bands_tv), filter_phase="linear"
        )

        self.noise_synth = libdamp.generators.SimpleFilteredNoise(
            frame_len, filt_len, fs, torch.linspace(0, fs / 2, num_bands_noise)
        )

    def estimate_params(self, x):
        """Estimate TI filter, TV filter, and noise synthesizer magnitudes from the LM input.
        """
        mel = torch.log(1 + self.mel_transform(x.unsqueeze(-2)))
        feat = self.control_cnn(mel)
        feat = self.control_rnn(feat)

        ti_mag = libdamp.exp_sigmoid(self.ti_head(feat.mean(dim=1, keepdim=True)) - 3, exp=math.log(10.0))
        noise_mag = libdamp.exp_sigmoid(self.noise_head(feat) - 3, exp=math.log(10.0))
        tv_mag = libdamp.exp_sigmoid(self.tv_head(feat) - 3, exp=math.log(10.0))

        return ti_mag, tv_mag, noise_mag

    def forward(self, x, return_intermediates=False):
        """Forward pass implementation.
        """
        ti_mag, tv_mag, noise_mag = self.estimate_params(x)

        self.ti_filter.clear()
        self.ti_filter.update(ti_mag)
        y_ti = self.ti_filter.process(x)

        y_norm = self.peak_norm(y_ti)

        noise = self.noise_synth.generate(noise_mag)
        y_noisy = y_norm + noise

        self.tv_filter.clear()
        self.tv_filter.update(tv_mag)
        y_hat = self.tv_filter.process(y_noisy)

        if return_intermediates:
            return y_hat, {
                "x_lm": x,
                "ti_filtered": y_ti,
                "peak_normed": y_norm,
                "noise": noise,
                "noisy": y_noisy,
                "y_hat": y_hat,
            }

        return y_hat

    def training_step(self, batch, _):
        x_lm, x_cm = batch

        y_hat = self(x_lm)
        loss = self.loss_fn(x_cm, y_hat).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        x_lm, x_cm = batch

        y_hat = self(x_lm)
        loss = self.loss_fn(x_cm, y_hat).mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
