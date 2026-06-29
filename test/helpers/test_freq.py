"""Unit tests for `libdamp.helpers.freq`."""

import torch

from libdamp.helpers.freq import hz2midi, midi2hz, timbre2harmonics


class TestHz2Midi:
    def test_a4_reference_maps_to_69(self):
        assert torch.isclose(hz2midi(440.0), torch.tensor(69.0))

    def test_octave_up_adds_twelve_semitones(self):
        assert torch.isclose(hz2midi(880.0), torch.tensor(81.0))

    def test_zero_hz_maps_to_zero_val(self):
        f = torch.tensor([0.0, 440.0])
        p = hz2midi(f, zero_val=-1)
        assert p[0] == -1
        assert torch.isclose(p[1], torch.tensor(69.0))

    def test_accepts_plain_float_input(self):
        # the type hint advertises float support; this used to crash (torch.ones_like on a float).
        p = hz2midi(440.0)
        assert torch.is_tensor(p)
        assert p.shape == ()

    def test_custom_reference_frequency(self):
        assert torch.isclose(hz2midi(415.0, a4_ref=415.0), torch.tensor(69.0))


class TestMidi2Hz:
    def test_midi_69_maps_to_440hz(self):
        assert torch.isclose(midi2hz(69.0), torch.tensor(440.0))

    def test_one_octave_doubles_frequency(self):
        assert torch.isclose(midi2hz(81.0), torch.tensor(880.0))

    def test_zero_val_maps_to_zero_hz(self):
        p = torch.tensor([-1.0, 69.0])
        f = midi2hz(p, zero_val=-1)
        assert f[0] == 0
        assert torch.isclose(f[1], torch.tensor(440.0))

    def test_accepts_plain_float_input(self):
        # the type hint advertises float support; this used to crash (item assignment on a float).
        f = midi2hz(69.0)
        assert torch.is_tensor(f)
        assert f.shape == ()

    def test_roundtrip_with_hz2midi(self):
        f = torch.tensor([220.0, 440.0, 880.0, 1234.5])
        assert torch.allclose(midi2hz(hz2midi(f)), f, rtol=1e-4)


class TestTimbre2Harmonics:
    def test_output_shape_is_h_by_2(self):
        A = timbre2harmonics("flat", 5)
        assert A.shape == (5, 2)

    def test_harmonic_factors_are_consecutive_integers_for_harmonic_timbres(self):
        A = timbre2harmonics("flat", 6)
        assert torch.equal(A[:, 0], torch.arange(1, 7, dtype=A.dtype))

    def test_flat_has_unit_amplitude_for_all_harmonics(self):
        A = timbre2harmonics("flat", 4)
        assert torch.all(A[:, 1] == 1)

    def test_square_has_zero_even_harmonics(self):
        A = timbre2harmonics("square", 6)
        assert torch.all(A[1::2, 1] == 0)
        assert A[0, 1] == 1

    def test_clarinet_like_has_zero_even_harmonics(self):
        A = timbre2harmonics("clarinet-like", 6)
        assert torch.all(A[1::2, 1] == 0)

    def test_sawtooth_fundamental_has_unit_amplitude(self):
        A = timbre2harmonics("sawtooth", 5)
        assert A[0, 1] == 1

    def test_random_inharmonic_perturbs_harmonics_above_the_fundamental(self):
        torch.manual_seed(0)
        A = timbre2harmonics("random_inharmonic", 5, harmonic_sigma=0.5)
        assert A[0, 0] == 1
        # with a large sigma, harmonics 2..H are extremely unlikely to land exactly on integers
        assert not torch.allclose(A[1:, 0], torch.arange(2, 6, dtype=A.dtype))

    def test_unknown_timbre_raises(self):
        try:
            timbre2harmonics("not-a-timbre", 4)
        except ValueError:
            pass
        else:
            assert False, "Expected a ValueError for an unknown timbre."
