"""Unit tests for `libdamp.datasets.combined`."""

import torch

from libdamp.datasets.combined import CombinedDataset, ZipDataset


class _ListDataset(torch.utils.data.Dataset):
    """Minimal dataset wrapping a Python list, for testing."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TestCombinedDataset:
    def test_length_is_sum_of_dataset_lengths(self):
        ds = CombinedDataset([_ListDataset([0, 1, 2]), _ListDataset([10, 11])], shuffle_items=False)
        assert len(ds) == 5

    def test_unshuffled_indices_walk_through_datasets_in_order(self):
        ds = CombinedDataset([_ListDataset([0, 1, 2]), _ListDataset([10, 11])], shuffle_items=False)
        assert [ds[i] for i in range(len(ds))] == [0, 1, 2, 10, 11]

    def test_shuffled_yields_same_multiset_of_items(self):
        torch.manual_seed(0)
        items_a = [0, 1, 2]
        items_b = [10, 11]
        ds = CombinedDataset([_ListDataset(items_a), _ListDataset(items_b)], shuffle_items=True)
        assert sorted(ds[i] for i in range(len(ds))) == sorted(items_a + items_b)

    def test_single_dataset(self):
        ds = CombinedDataset([_ListDataset([0, 1, 2])], shuffle_items=False)
        assert len(ds) == 3
        assert [ds[i] for i in range(len(ds))] == [0, 1, 2]


class TestZipDataset:
    def test_length_matches_dataset_length(self):
        ds = ZipDataset([_ListDataset([0, 1, 2]), _ListDataset([10, 11, 12])], shuffle_items=False)
        assert len(ds) == 3

    def test_unshuffled_assignment_is_aligned_by_index(self):
        ds = ZipDataset([_ListDataset([0, 1, 2]), _ListDataset([10, 11, 12])], shuffle_items=False)
        assert [ds[i] for i in range(len(ds))] == [[0, 10], [1, 11], [2, 12]]

    def test_shuffled_assignment_is_still_a_valid_permutation_per_dataset(self):
        torch.manual_seed(0)
        items_a = [0, 1, 2, 3]
        items_b = [10, 11, 12, 13]
        ds = ZipDataset([_ListDataset(items_a), _ListDataset(items_b)], shuffle_items=True)
        seen_a = [ds[i][0] for i in range(len(ds))]
        seen_b = [ds[i][1] for i in range(len(ds))]
        assert sorted(seen_a) == sorted(items_a)
        assert sorted(seen_b) == sorted(items_b)

    def test_mismatched_dataset_lengths_raise(self):
        try:
            ZipDataset([_ListDataset([0, 1]), _ListDataset([0, 1, 2])], shuffle_items=False)
        except AssertionError:
            pass
        else:
            assert False, "Expected an AssertionError for mismatched dataset lengths."
