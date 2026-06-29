"""Unit tests for `libdamp.generators.impulse_train`."""

import torch

from libdamp.generators.impulse_train import ImpulseTrain


class TestImpulseTrain:
    def test_output_shape_with_sum_up(self):
        gen = ImpulseTrain(N=64, M=8, fs=16000.0, sum_up=True)
        gen.update(f0=torch.full((1, 4), 220.0))
        out = gen.generate()
        assert out.shape == (1, 64 * 4)

    def test_output_shape_without_sum_up(self):
        gen = ImpulseTrain(N=64, M=8, fs=16000.0, sum_up=False)
        gen.update(f0=torch.full((1, 4), 220.0))
        out = gen.generate()
        assert out.shape == (1, 8, 64 * 4)

    def test_output_is_normalized_to_unit_peak(self):
        gen = ImpulseTrain(N=64, M=8, fs=16000.0)
        gen.update(f0=torch.full((1, 4), 220.0))
        out = gen.generate()
        assert torch.isclose(out.abs().max(), torch.tensor(1.0), atol=1e-3)

    def test_inharmonicity_changes_output(self):
        gen_a = ImpulseTrain(N=64, M=8, fs=16000.0)
        gen_a.update(f0=torch.full((1, 4), 220.0))
        x_a = gen_a.generate()

        gen_b = ImpulseTrain(N=64, M=8, fs=16000.0)
        inharmonicity = torch.ones(1, 8, 4)
        inharmonicity[:, 1:, :] = 1.02
        gen_b.update(f0=torch.full((1, 4), 220.0), inharmonicity=inharmonicity)
        x_b = gen_b.generate()

        assert not torch.allclose(x_a, x_b)

    def test_clear_delegates_to_underlying_oscillator(self):
        gen = ImpulseTrain(N=64, M=8, fs=16000.0)
        gen.update(f0=torch.full((1, 4), 220.0))
        gen.generate()
        gen.clear()
        assert gen.osc.initialized is False
