"""Learn sinusoidal + noise synthesis for VocalRevolutions"""

import math

import gin
import torch
import torchaudio as taudio

import libdamp


@gin.configurable
class SinesAndNoiseExperiment(libdamp.Experiment):
    """Learn sinusoidal + noise synthesis for VocalRevolutions"""

    def __init__(
        self,
        N=256,
        fs=44100.0,
        learning_rate=1e-4,
        loss_fn=None,
        num_sines=24,
        num_freq_bins=1,
        num_noise_bands=8,
        n_fft_mels=4096,
        f_min_mels=100.0,
        f_max_mels=5000.0,
        n_mels=256,
        model_size=512,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        if loss_fn is None:
            self.loss_fn = libdamp.MSSLoss()
        else:
            self.loss_fn = loss_fn

        self.learning_rate = learning_rate

        self.num_sines = num_sines
        self.num_freq_bins = num_freq_bins
        self.num_noise_bands = num_noise_bands

        self.meltransform = taudio.transforms.MelSpectrogram(
            fs, n_fft=n_fft_mels, hop_length=N, f_min=f_min_mels, f_max=f_max_mels, n_mels=n_mels, center=False, pad=(n_fft_mels - N) // 2
        )

        self.sine_synth = libdamp.generators.SinusoidalOsc(N, fs, interp_f="end_linear", interp_a="end_linear")
        self.band_synth = libdamp.generators.BandFilteredNoise(N, self.num_noise_bands, 2, fs)

        self.freq_scaling = libdamp.LogitsToFreq(bins_per_freq=self.num_freq_bins, f_min=50, f_max=12000)
        self.fc_scaling = libdamp.LogitsToFreq(bins_per_freq=self.num_freq_bins, f_min=200, f_max=15000)

        self.model = torch.nn.Sequential(
            libdamp.ConvStack(n_mels, model_size),
            libdamp.BiLSTM(model_size, model_size // 2),
        )

        self.sine_f_head = torch.nn.Linear(model_size, self.num_freq_bins * self.num_sines)
        self.sine_a_head = torch.nn.Linear(model_size, self.num_sines)

        if self.num_noise_bands > 0:
            self.noise_fc_head = torch.nn.Linear(model_size, self.num_freq_bins * self.num_noise_bands)
            self.noise_bw_head = torch.nn.Linear(model_size, self.num_noise_bands)
            self.noise_ba_head = torch.nn.Linear(model_size, self.num_noise_bands)

        self.env_head = torch.nn.Linear(model_size, 1)
        self.env = libdamp.processors.GainEnvelope(interp_mode="end_linear")

        self.loss_fn = loss_fn

    def estimate_params(self, x):
        x = x.unsqueeze(1)  # add channel dim
        X = torch.log(1 + self.meltransform(x))
        z = self.model(X)

        f_s = self.sine_f_head(z)
        f_s = self.freq_scaling(f_s)
        f_s = torch.transpose(f_s, -2, -1)  # shape: (B, N, F)
        # f_s *= 2 ** torch.linspace(0, 5.6, self.num_sines)[None, :, None].to(f_s)  # distribute in octaves
        # f_s += torch.linspace(0)[None, :, None].to(f_s) # linear bias

        a_s = self.sine_a_head(z)
        a_s = libdamp.exp_sigmoid(a_s - 3, exp=math.log(10.0))
        a_s = torch.transpose(a_s, -2, -1)  # shape: (B, N, F)

        g = self.env_head(z)
        g = libdamp.exp_sigmoid(g, exp=math.log(10.0)).squeeze(-1)  # shape: (B, F)

        if self.num_noise_bands > 0:
            fc = self.noise_fc_head(z)
            fc = self.fc_scaling(fc)  # shape: (B, F, N)

            q = self.noise_bw_head(z)
            q = libdamp.exp_sigmoid(q, exp=math.log(10.0)) / 2.05 + 0.01
            bw = q * fc.detach()  # shape: (B, F, N)

            ba = self.noise_ba_head(z)
            ba = libdamp.exp_sigmoid(ba - 3, exp=math.log(10.0))  # shape: (B, F, N)
        else:
            fc = None
            bw = None
            ba = None

        return g, f_s, a_s, fc, bw, ba

    def synth_signal(self, g, f_s, a_s, fc, bw, ba, sum_up=True):
        self.sine_synth.clear()
        self.sine_synth.update(f_s, a_s)
        y_s = self.sine_synth.generate(sum_up=sum_up)

        y = y_s

        if self.num_noise_bands > 0:
            self.band_synth.clear()
            self.band_synth.update(fc, bw, ba)
            y_n = self.band_synth.generate(sum_up=sum_up)
            if not sum_up:
                y = torch.concat([y, y_n], dim=1)
            else:
                y = y + y_n

        self.env.clear()
        self.env.update(g)

        return self.env.process(y)

    def forward(self, x, sum_up=True, return_f=False):
        g, f_s, a_s, fc, bw, ba = self.estimate_params(x)
        y = self.synth_signal(g, f_s, a_s, fc, bw, ba, sum_up)

        if return_f:
            return y, f_s

        return y

    def training_step(self, batch, _):
        x, f0 = batch
        y, f_s = self(x, return_f=True)

        f_gt = f0[:,None,:] * torch.arange(1, self.num_sines+1).to(f0.device)[None,:,None]
        mask = (f_gt < 0)
        loss_f = 0.001 * ((f_s[mask] - f_gt[mask])**2).mean()
        loss_r = self.loss_fn(x.squeeze(), y.squeeze()).mean()

        loss = loss_r + loss_f
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("rec_loss", loss_r, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("freq_loss", loss_f, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        x, _ = batch
        y = self(x)

        loss = self.loss_fn(x.squeeze(), y.squeeze()).mean()
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, _):
        x, _ = batch
        y = self(x)

        loss = self.loss_fn(x.squeeze(), y.squeeze()).mean()
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
