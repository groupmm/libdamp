"""Dataset wrapper for combining multiple other datasets."""

import gin
import numpy as np
import torch


@gin.register
class CombinedDataset(torch.utils.data.Dataset):
    """Dataset wrapper for combining multiple existing datasets"""

    def __init__(self, datasets: list[torch.utils.data.Dataset] = [], shuffle_items: bool = True):
        """Dataset wrapper for combining multiple existing datasets.

        Parameters
        ----------
        datasets : list of torch.utils.data.Dataset
            Instances of datasets to be combined.
        shuffle_items : bool
            Whether the items should be shuffled across datasets (default: True).
        """

        self.datasets = datasets

        L = []
        for ds in self.datasets:
            L.append(len(ds))

        self.last_idx = np.cumsum(L)

        self.shuffle_items = shuffle_items
        if self.shuffle_items:
            self.shuffle = np.random.permutation(self.last_idx[-1])

    def __len__(self) -> int:
        return self.last_idx[-1]

    def __getitem__(self, idx: int):
        """Get a sample from the combined datasets.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve, in the combined index range
            `[0, len(self))`.

        Returns
        -------
        Whatever the underlying dataset returns for the selected sample.
        """
        if self.shuffle_items:
            i = self.shuffle[idx]
        else:
            i = idx
        j = np.searchsorted(self.last_idx - 1, i)  # which dataset to use
        k = i - self.last_idx[j - 1] if j >= 1 else i  # which index in the dataset
        return self.datasets[j][k]


@gin.register
class ZipDataset(torch.utils.data.Dataset):
    """Dataset wrapper for retrieving items from multiple datasets simultaneously (like Python `zip`)"""

    def __init__(self, datasets: list[torch.utils.data.Dataset] = [], shuffle_items: bool = True) -> None:
        """Dataset wrapper for retrieving items from multiple datasets simultaneously (like Python `zip`)


        Parameters
        ----------
        datasets : list of torch.utils.data.Dataset
            Instances of datasets to be combined.
        shuffle_items : bool
            Whether items should be combined randomly or with fixed assignment (item 0 is item 0
            from all datasets) (default: True).
        """

        self.datasets = datasets

        self.L = None
        self.assignments = []

        for ds in self.datasets:
            if self.L is None:
                self.L = len(ds)
            else:
                assert len(ds) == self.L, "Datasets must all have the same length."

            if shuffle_items:
                self.assignments.append(np.random.permutation(self.L))
            else:
                self.assignments.append(np.arange(self.L))

    def __len__(self) -> int:
        return self.L

    def __getitem__(self, idx: int) -> list:
        """Get aligned samples from all zipped datasets.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        list
            List of samples, one from each dataset.
        """
        output = []
        for i, ds in enumerate(self.datasets):
            idx_ = self.assignments[i][idx]
            output.append(ds[idx_])

        return output
