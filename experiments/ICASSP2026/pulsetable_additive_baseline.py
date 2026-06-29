"""Additive synthesis baseline for comparison with the pulsetable method"""

import math

import gin
import torch
import torchaudio as taudio

import libdamp


class PostFilter(torch.nn.Module):
    """Fixed FIR post-filter with learnable coefficients"""

    def __init__(self, N, L, initializiation="unity"):
        super().__init__()

        if initializiation == "unity":
            initial_filter = torch.zeros((1, 1, L))
            initial_filter[..., 0] = 1
        else:
            raise ValueError(f"Unknown initialization type '{initializiation}'.")

        self.conv = libdamp.processors.TVFIRFilter(N, L)

        self.register_parameter("filter", torch.nn.Parameter(initial_filter))

    def forward(self, x):
        """Forward pass implementation"""
        self.conv.clear()
        self.conv.update(self.filter)
        return self.conv.process(x)


@gin.configurable
class AdditivePulseTableExperiment(libdamp.Experiment):
    """Additive synthesis baseline for comparison with the pulsetable method"""

    def __init__(
        self,
        N=256,
        fs=48000.0,
        learning_rate=1e-4,
        loss_fn=None,
        num_harm=100,
        learn_gain=True,
        n_fft_mels=4096,
        f_min_mels=100.0,
        f_max_mels=5000.0,
        n_mels=256,
        model_size_weights=32,
        model_size_gain=32,
        model_size_freq=32,
        model_size_noise=32,
        w_entropy=0.1,
        w_sparsity=0.1,
        learn_noise=False,
        learn_post_filter=False,
        learn_freq_offset=False,
        envelope_oversampling=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if loss_fn is None:
            self.loss_fn = libdamp.MSSLoss()
        else:
            self.loss_fn = loss_fn

        self.fs = fs
        self.N = N
        self.num_harm = num_harm

        self.learn_gain = learn_gain
        self.learn_noise = learn_noise
        self.learn_post_filter = learn_post_filter
        self.learn_freq_offset = learn_freq_offset

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.w_entropy = w_entropy
        self.w_sparsity = w_sparsity

        self.envelope_oversampling = envelope_oversampling

        N_synth = N if not self.learn_freq_offset else N // self.envelope_oversampling
        self.synth = libdamp.generators.HarmonicOsc(N_synth, self.fs, interp_f="end_linear", interp_a="end_linear")

        self.g_scaling = libdamp.FreqToBins(num_bins=8, f_min=1e-2, f_max=1, target_smoothing=600)

        self.f0_scaling = libdamp.FreqToBins(num_bins=8, f_min=150, f_max=1500, target_smoothing=300)

        self.weight_mapping = torch.nn.Sequential(
            torch.nn.Linear(16, model_size_weights),
            torch.nn.ReLU(),
            torch.nn.Linear(model_size_weights, self.num_harm),
        )

        if self.learn_gain or self.learn_noise:
            self.meltransform = taudio.transforms.MelSpectrogram(
                self.fs, n_fft=n_fft_mels, hop_length=N, f_min=f_min_mels, f_max=f_max_mels, n_mels=n_mels, center=False, pad=(n_fft_mels - N) // 2
            )

        if self.learn_gain:
            # code for model adapted from
            # https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/transcriber.py
            self.gain_model = torch.nn.Sequential(
                libdamp.ConvStack(n_mels, model_size_gain),
                libdamp.BiLSTM(model_size_gain, model_size_gain // 2),
                torch.nn.Linear(model_size_gain, self.envelope_oversampling),
            )

        if self.learn_freq_offset:
            self.freq_model = torch.nn.Sequential(
                libdamp.ConvStack(n_mels, model_size_freq),
                libdamp.BiLSTM(model_size_freq, model_size_freq // 2),
                torch.nn.Linear(model_size_freq, self.envelope_oversampling),
                torch.nn.Tanh(),
            )

        if self.learn_noise:
            L_noise = 256
            noise_center_freqs = torch.fft.rfftfreq(L_noise, 1 / self.fs)  # libdamp.THIRD_OCTAVE_BANDS
            self.noise_model = torch.nn.Sequential(
                libdamp.ConvStack(n_mels, model_size_noise),
                libdamp.BiLSTM(model_size_noise, model_size_noise // 2),
                torch.nn.Linear(model_size_noise, len(noise_center_freqs)),
            )
            self.noise_synth = libdamp.generators.SimpleFilteredNoise(N, L_noise, self.fs, noise_center_freqs)

        if self.learn_post_filter:
            self.post_filter = PostFilter(N, 4096)

    def estimate_weights(self, f0, g):
        """Estimate table weighting from parameter from an F0 + gain trajectory"""
        f0_sc = self.f0_scaling(f0)
        g_sc = self.g_scaling(g)
        inp = torch.cat([f0_sc, g_sc], dim=-1)
        w = self.weight_mapping(inp)

        # scale and normalize amplitudes
        w = libdamp.exp_sigmoid(w - 3, exp=math.log(10.0))
        w /= torch.sum(w, dim=-1, keepdims=True)

        return w

    def estimate_gain(self, x):
        """Estimate gain parameter from input audio"""
        B = x.shape[0]  # batch size
        X = torch.log(1 + self.meltransform(x.unsqueeze(-2)))
        g_est = self.gain_model(X)
        g_est = libdamp.exp_sigmoid(g_est.reshape(B, -1) - 3, exp=math.log(10.0))  # 10**((g_est.reshape(B, -1) - 3)/2)

        return g_est

    def estimate_freq_offset(self, x, f0):
        """Estimate F0 offset from input audio"""
        B = x.shape[0]  # batch size
        X = torch.log(1 + self.meltransform(x.unsqueeze(-2)))
        f0_offs = self.freq_model(X)
        f0_ext = libdamp.interpolate_samples(f0, self.envelope_oversampling, mode="const")

        return f0_ext * (2 ** (20 / 1200 * f0_offs.reshape(B, -1)))

    def estimate_noise(self, x):
        """Estimate noise component from input audio"""
        X = torch.log(1 + self.meltransform(x.unsqueeze(-2)))
        n = self.noise_model(X)
        n = libdamp.exp_sigmoid(n - 3, exp=math.log(10.0))
        n = n.squeeze(2)  # remove channel dim
        return self.noise_synth.generate(n)

    def synth_add(self, f0, g, w):
        """Synthesize a tonal signal additively"""
        self.synth.clear()

        a = g[..., None] * w
        self.synth.update(f0, a.transpose(-2, -1))

        return self.synth.generate()

    def forward(self, x, f0, g, growl=None, return_weight=False, return_pre_filt=False):
        if self.learn_gain:
            g_est = self.estimate_gain(x)
        else:
            g_est = g

        if self.learn_freq_offset:
            f0_est = self.estimate_freq_offset(x, f0)
        else:
            f0_est = f0

        if growl is not None:
            phi = torch.cumsum(torch.ones_like(f0_est) * 160 / self.N, dim=-1)
            f0_est += 20 * growl * torch.sin(2 * torch.pi * phi)

        if self.learn_freq_offset:
            w = self.estimate_weights(f0_est.detach(), g_est.detach())
        else:
            w = self.estimate_weights(f0_est, g_est.detach()[:, :: self.envelope_oversampling])

        if self.learn_post_filter:
            y_pre = self.synth_add(f0_est, g_est, w)
            if self.learn_noise and (not self.training or self.current_epoch > 3):
                y_pre += self.estimate_noise(x)
            y = self.post_filter(y_pre)
        else:
            y_pre = None
            y = self.synth_add(f0_est, g_est, w)
            if self.learn_noise and (not self.training or self.current_epoch > 3):
                y += self.estimate_noise(x)

        returns = [y]
        if return_weight:
            returns.append(w)
        if return_pre_filt:
            returns.append(y_pre)

        if len(returns) == 1:
            return returns[0]

        return returns

    def training_step(self, batch, _):
        x, f0, _, g = batch

        y, _, y_pre = self(x, f0, g, return_weight=True, return_pre_filt=True)

        if y_pre is not None:
            L_mss_pre = self.loss_fn(x, y_pre).mean()
            L_mss_pst = self.loss_fn(x, y).mean()
            L_mss = 0.5 * L_mss_pre + 0.5 * L_mss_pst
            self.log("L_mss_pre", L_mss_pre, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("L_mss_pst", L_mss_pst, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            L_mss = self.loss_fn(x, y).mean()

        # can't use the entropy loss here
        L_ent = 0

        if self.w_sparsity > 0 and self.learn_post_filter:
            L_spf = torch.linalg.norm(self.post_filter.filter, ord=1, dim=-1).mean()
        else:
            L_spf = 0

        loss = L_mss + self.w_entropy * L_ent + self.w_sparsity * L_spf

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("L_mss", L_mss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("L_ent", L_ent, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log("L_spf", L_spf, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, _):
        x, f0, _, g = batch

        y = self(x, f0, g)

        L_mss = self.loss_fn(x, y).mean()

        self.log("val_loss", L_mss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return L_mss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

        return [optimizer], [scheduler]
