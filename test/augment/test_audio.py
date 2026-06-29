"""Unit tests for `libdamp.augment.audio`."""

import torch

from libdamp.augment.audio import (
    DropoutArtifactAugmentation,
    NoiseArtifactAugmentation,
    NoiseAugmentation,
)


def _rms(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2, dim=dim))


class TestNoiseAugmentation:
    def test_output_shape_matches_input_shape(self):
        aug = NoiseAugmentation(prob=1.0)
        inp = torch.randn(3, 500)
        out = aug(inp)
        assert out.shape == inp.shape

    def test_zero_db_noise_level_matches_signal_rms(self):
        """At noise_level_db=0 the RMS of the injected noise should equal the RMS of the input."""
        torch.manual_seed(0)
        aug = NoiseAugmentation(prob=1.0, noise_level_db=0.0)
        inp = torch.randn(1, 200_000)
        out = aug.apply(inp)
        noise = out - inp

        a_inp = _rms(inp, dim=-1)
        a_noise = _rms(noise, dim=-1)
        assert torch.allclose(a_noise, a_inp, rtol=0.02)

    def test_minus_10_db_noise_level_is_attenuated_by_10_db(self):
        torch.manual_seed(0)
        aug = NoiseAugmentation(prob=1.0, noise_level_db=-10.0)
        inp = torch.randn(1, 200_000)
        out = aug.apply(inp)
        noise = out - inp

        a_inp = _rms(inp, dim=-1)
        a_noise = _rms(noise, dim=-1)
        expected_ratio = 10 ** (-10.0 / 20)
        assert torch.allclose(a_noise / a_inp, torch.tensor(expected_ratio), rtol=0.02)

    def test_noise_is_applied_independently_per_batch_item(self):
        torch.manual_seed(0)
        aug = NoiseAugmentation(prob=1.0, noise_level_db=0.0)
        inp = torch.ones(2, 1000)
        out = aug.apply(inp)
        # the two batch items use independent random draws, so they should not be identical
        assert not torch.equal(out[0], out[1])


class TestDropoutArtifactAugmentation:
    def test_output_shape_matches_input_shape(self):
        aug = DropoutArtifactAugmentation(prob=1.0, min_len=5, max_len=10)
        inp = torch.randn(3, 100)
        out = aug.apply(inp)
        assert out.shape == inp.shape

    def test_introduces_a_silent_segment_of_expected_length(self):
        aug = DropoutArtifactAugmentation(prob=1.0, min_len=10, max_len=10)
        inp = torch.ones(4, 200)
        out = aug.apply(inp)

        for b in range(inp.shape[0]):
            zero_mask = out[b] == 0
            assert zero_mask.sum() == 10
            # the zeroed samples must form one contiguous block
            nonzero_idx = zero_mask.nonzero(as_tuple=True)[0]
            assert torch.equal(nonzero_idx, torch.arange(nonzero_idx[0], nonzero_idx[0] + 10))

    def test_does_not_modify_input_tensor(self):
        aug = DropoutArtifactAugmentation(prob=1.0, min_len=5, max_len=10)
        inp = torch.randn(2, 100)
        inp_copy = inp.clone()
        aug.apply(inp)
        assert torch.equal(inp, inp_copy)


class TestNoiseArtifactAugmentation:
    def test_output_shape_matches_input_shape(self):
        aug = NoiseArtifactAugmentation(prob=1.0, min_len=10, max_len=50)
        inp = torch.zeros(3, 2000)
        out = aug.apply(inp)
        assert out.shape == inp.shape

    def test_only_a_bounded_segment_is_modified(self):
        aug = NoiseArtifactAugmentation(prob=1.0, min_len=20, max_len=20, min_ampl_db=0, max_ampl_db=0)
        inp = torch.zeros(2, 2000)
        out = aug.apply(inp)

        for b in range(inp.shape[0]):
            changed = (out[b] != 0).nonzero(as_tuple=True)[0]
            assert changed.numel() == 20
            assert torch.equal(changed, torch.arange(changed[0], changed[0] + 20))

    def test_added_noise_respects_amplitude_bounds(self):
        torch.manual_seed(0)
        max_ampl_db = -6.0
        aug = NoiseArtifactAugmentation(prob=1.0, min_len=1000, max_len=1000, min_ampl_db=max_ampl_db, max_ampl_db=max_ampl_db)
        inp = torch.zeros(1, 2000)
        out = aug.apply(inp)
        max_ampl = 10 ** (max_ampl_db / 20)
        assert torch.all(out.abs() <= max_ampl + 1e-6)

    def test_does_not_modify_input_tensor(self):
        aug = NoiseArtifactAugmentation(prob=1.0, min_len=10, max_len=50)
        inp = torch.randn(2, 2000)
        inp_copy = inp.clone()
        aug.apply(inp)
        assert torch.equal(inp, inp_copy)
