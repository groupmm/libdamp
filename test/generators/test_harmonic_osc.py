"""Unit tests for `libdamp.generators.harmonic_osc`."""

import torch

from libdamp.generators.harmonic_osc import HarmonicOsc


class TestHarmonicOsc:
    def test_output_shape(self):
        osc = HarmonicOsc(N=64, fs=16000.0)
        osc.update(f0=torch.full((2, 4), 110.0), a=torch.ones(2, 3, 4))
        out = osc.generate()
        assert out.shape == (2, 64 * 4)

    def test_harmonics_are_integer_multiples_of_f0(self):
        fs = 16000.0
        N = 256
        f0 = 100.0
        osc = HarmonicOsc(N=N, fs=fs)

        a = torch.zeros(1, 3, 2)
        a[:, 0, :] = 1.0
        osc.update(f0=torch.full((1, 2), f0), a=a)
        out = osc.generate()

        t = torch.arange(1, out.shape[-1] + 1)
        expected = torch.sin(2 * torch.pi * f0 * t / fs)
        assert torch.allclose(out[0], expected, atol=1e-3)

    def test_inharmonicity_shifts_harmonic_frequencies(self):
        fs = 16000.0
        N = 256
        f0 = 100.0
        osc_harm = HarmonicOsc(N=N, fs=fs)
        osc_inharm = HarmonicOsc(N=N, fs=fs)

        a = torch.ones(1, 2, 2)
        osc_harm.update(f0=torch.full((1, 2), f0), a=a)
        x_harm = osc_harm.generate()

        inharmonicity = torch.ones(1, 2, 2)
        inharmonicity[:, 1, :] = 1.01  # detune the 2nd harmonic
        osc_inharm.update(f0=torch.full((1, 2), f0), a=a, inharmonicity=inharmonicity)
        x_inharm = osc_inharm.generate()

        assert not torch.allclose(x_harm, x_inharm)

    def test_clear_delegates_to_underlying_oscillator(self):
        osc = HarmonicOsc(N=64, fs=16000.0)
        osc.update(f0=torch.full((1, 2), 110.0), a=torch.ones(1, 1, 2))
        osc.generate()
        osc.clear()
        assert osc.osc.initialized is False
