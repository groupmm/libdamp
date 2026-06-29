"""Wind instrument synthesis with the pulsetable method"""

import math

import gin
import torch
import torchaudio as taudio

import libdamp


class WeightedPulseTableSynthesizer(torch.nn.Module):
    """Wrapper for the full synthesis pipeline, consisting of a pulsetable, low-pass filter, and envelope"""

    def __init__(self, mode, table, table_mask, fs, N):
        super().__init__()

        self.pulsetable = libdamp.generators.WeightedTableOsc(
            N, table, fs, mode=mode, learnable_table=True, interp_f0="end_linear", interp_w="end_linear", table_mask=table_mask
        )

        self.envelope = libdamp.processors.Envelope(interp_mode="end_linear")

    def forward(self, f0, g, w):
        """Forward pass implementation"""
        self.pulsetable.clear()
        self.pulsetable.update(f0, w)
        x = self.pulsetable.generate()

        self.envelope.clear()
        self.envelope.update(g)
        x = self.envelope.process(x).squeeze(1)  # remove channel dim

        return x


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
class PulseTableExperiment(libdamp.Experiment):
    """Instrument Resynthesis with the pulsetable method"""

    def __init__(
        self,
        N=256,
        fs=48000.0,
        learning_rate=1e-4,
        loss_fn=None,
        L_proto=64,
        pulse_init="quadratic_decay",
        num_pulses=1,
        mask_pulses=False,
        table_mode="pulse",
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

        self.example_input_array = (
            torch.randn(4, 192000),  # x (B, N*F)
            torch.randn(4, 1500),  # f0 (B, F)
            torch.randn(4, 1500),  # g (B, F)
        )

        self.fs = fs
        self.L_proto = L_proto
        self.N = N
        self.num_pulses = num_pulses

        self.learn_gain = learn_gain
        self.learn_noise = learn_noise
        self.learn_post_filter = learn_post_filter
        self.learn_freq_offset = learn_freq_offset

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.w_entropy = w_entropy
        self.w_sparsity = w_sparsity

        self.envelope_oversampling = envelope_oversampling

        if pulse_init == "tp":
            initial_table = torch.zeros((self.num_pulses, self.L_proto))

            # fmt: off
            # Bb Trumpet G5 pulse from Christian
            initial_table[:, :62] = torch.tensor([0.00320435, 0.02990723, 0.05715942,
                    0.08157349, 0.10101318, 0.11645508,
                    0.13339233, 0.15710449, 0.19476318,
                    0.25909424, 0.37286377, 0.53939819,
                    0.69140625, 0.76544189, 0.75384521,
                    0.65542603, 0.47903442, 0.26800537,
                    0.06671143, -0.11914062, -0.29125977,
                    -0.43798828, -0.54736328, -0.61621094,
                    -0.6506958, -0.65618896, -0.63525391,
                    -0.58712769, -0.51498413, -0.4284668,
                    -0.33883667, -0.25408936, -0.17715454,
                    -0.10717773, -0.04232788, 0.01751709,
                    0.07049561, 0.11480713, 0.15014648,
                    0.17633057, 0.19268799, 0.19772339,
                    0.19073486, 0.17276001, 0.14749146,
                    0.11923218, 0.0894165, 0.05593872,
                    0.0171814, -0.02511597, -0.06549072,
                    -0.09692383, -0.11602783, -0.12533569,
                    -0.12841797, -0.12487793, -0.11401367,
                    -0.09667969, -0.07543945, -0.05270386,
                    -0.02957153, -0.0050354])[None, :]
            # fmt: on
        elif pulse_init == "noise":
            random_table = torch.randn(self.num_pulses, self.L_proto)  # initialize with noise
            initial_table = random_table / torch.max(torch.abs(random_table), dim=-1, keepdim=True).values
        elif pulse_init == "quadratic_decay":
            # design a pulse with a quadratically decaying magnitude response
            P_ = torch.linspace(1, 0, (self.L_proto + 2) // 2) ** 2
            P_[0] = 0
            f_ = torch.fft.rfftfreq(self.L_proto, 1 / fs)
            proto_pulse = libdamp.design_fir_filter(f_, P_, self.fs, self.L_proto, phase="minimum")
            proto_pulse /= torch.max(torch.abs(proto_pulse))

            initial_table = proto_pulse[None, :].repeat(self.num_pulses, 1)
        else:
            raise ValueError(f"Unknown pulse initialization type '{pulse_init}'.")

        if mask_pulses:
            quarter = self.num_pulses // 4
            table_mask = torch.ones((self.num_pulses, self.L_proto))
            table_mask[1 * quarter : 2 * quarter, -self.L_proto // 4 * 1 :] = 0
            table_mask[2 * quarter : 3 * quarter, -self.L_proto // 4 * 2 :] = 0
            table_mask[3 * quarter : 4 * quarter, -self.L_proto // 4 * 3 :] = 0
        else:
            table_mask = None

        N_synth = N if not self.learn_freq_offset else N // self.envelope_oversampling
        self.synth = WeightedPulseTableSynthesizer(table_mode, initial_table, table_mask, self.fs, N_synth)

        self.g_scaling = libdamp.FreqToBins(num_bins=8, f_min=1e-2, f_max=1, target_smoothing=600)

        self.f0_scaling = libdamp.FreqToBins(num_bins=8, f_min=150, f_max=1500, target_smoothing=300)

        self.weight_mapping = torch.nn.Sequential(
            torch.nn.Linear(16, model_size_weights),
            torch.nn.ReLU(),
            torch.nn.Linear(model_size_weights, self.num_pulses),
        )
        self.gumbel_min_temp_exp = -2
        self.gumbel_max_temp_exp = 0
        self.gumbel_annealing_epochs = 30

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

        # gumbel softmax with temperature annealing
        if self.training and self.current_epoch < self.gumbel_annealing_epochs:
            exp = (self.gumbel_max_temp_exp - self.gumbel_min_temp_exp) * (
                1 - self.current_epoch / self.gumbel_annealing_epochs
            ) + self.gumbel_min_temp_exp
            tau = 10**exp
        else:
            tau = 10**self.gumbel_min_temp_exp

        w = torch.nn.functional.softmax(w / tau, dim=-1)

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
            # Add some vibrato (frequency modulation)
            phi = torch.cumsum(torch.ones_like(f0_est) * 160 / self.N, dim=-1)
            f0_est += 20 * growl * torch.sin(2 * torch.pi * phi)

        if self.learn_freq_offset:
            w = self.estimate_weights(f0_est.detach(), g_est.detach())
        else:
            w = self.estimate_weights(f0_est, g_est.detach()[:, :: self.envelope_oversampling])

        if self.learn_post_filter:
            y_pre = self.synth(f0_est, g_est, w)
            if self.learn_noise and (not self.training or self.current_epoch > 3):
                y_pre += self.estimate_noise(x)
            y = self.post_filter(y_pre)
        else:
            y_pre = None
            y = self.synth(f0_est, g_est, w)
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

        y, w, y_pre = self(x, f0, g, return_weight=True, return_pre_filt=True)

        if y_pre is not None:
            L_mss_pre = self.loss_fn(x, y_pre).mean()
            L_mss_pst = self.loss_fn(x, y).mean()
            L_mss = 0.5 * L_mss_pre + 0.5 * L_mss_pst
            self.log("L_mss_pre", L_mss_pre, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("L_mss_pst", L_mss_pst, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            L_mss = self.loss_fn(x, y).mean()

        # we want to maximize the entropy of the average usage of different pulses over a batch
        w_avg = torch.mean(w, dim=(0, 1))  # averaged over batch and time dimension
        neg_entropy = torch.sum(w_avg * torch.log(w_avg + 1e-10))
        L_ent = neg_entropy - math.log(1 / self.num_pulses)  # ensure positive loss value

        if self.w_sparsity > 0 and self.learn_post_filter:
            L_spf = torch.linalg.norm(self.post_filter.filter, ord=1, dim=-1).mean()
        else:
            L_spf = 0

        loss = L_mss + self.w_entropy * L_ent + self.w_sparsity * L_spf

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("L_mss", L_mss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("L_ent", L_ent, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log("L_spf", L_spf, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log("perplexity", torch.exp(-neg_entropy), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f0, _, g = batch

        y = self(x, f0, g)

        L_mss = self.loss_fn(x, y).mean()

        self.log("val_loss", L_mss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_audio(y, reference=x, current_batch_idx=batch_idx, split="val")

        return L_mss

    def test_step(self, batch, _):
        x, f0, _, g = batch

        y = self(x, f0, g)

        L_mss = self.loss_fn(x, y).mean()

        self.log("test_loss", L_mss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return L_mss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

        return [optimizer], [scheduler]
