import numpy as np

from operator import itemgetter
from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Optional


# From https://github.com/catalyst-team/catalyst
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class Subsampler(Sampler):
    """
    This sampler is useful if you have a very large dataset, and you don't want
    to define an epoch as a single pass through the dataset. 
    """
    def __init__(self, dataset, N_sample):
        super().__init__(data_source=dataset)
        assert len(dataset) > N_sample, f'`N_sample` {N_sample} should be less than length of dataset {len(dataset)}'
        self.len_dataset = len(dataset)
        self.N_sample = N_sample
        self.available_indices = list(range(len(dataset)))

    def __iter__(self):
        if len(self.available_indices) >= self.N_sample:
            subsampled = np.random.choice(self.available_indices, self.N_sample, replace=False)
            self.available_indices = list(set(self.available_indices) - set(subsampled))
        else:
            subsampled = list(self.available_indices) 
            self.available_indices = list(set(range(self.len_dataset)) - set(subsampled))
            N_remaining = self.N_sample - len(subsampled)
            remainder = list(np.random.choice(self.available_indices, N_remaining, replace=False))
            subsampled.extend(remainder)
            self.available_indices = list(set(self.available_indices) - set(remainder))
        assert len(subsampled) == self.N_sample
        return iter(subsampled)

    def __len__(self):
        return self.N_sample


class BalancedSampler(Sampler):
    """
    This sampler is for balanced sampling specifically for binary classification
    tasks.
    """
    def __init__(self, dataset, N_sample, pos_frac=0.5):
        super().__init__(data_source=dataset)
        self.len_dataset = len(dataset)
        self.negatives = np.where(np.asarray(dataset.labels) < 0.5)[0]
        self.positives = np.where(np.asarray(dataset.labels) >= 0.5)[0]
        self.pos_N = int(N_sample * pos_frac)
        self.neg_N = int(N_sample * (1 - pos_frac))

    def __iter__(self):
        negatives = np.random.choice(self.negatives, self.neg_N, replace=len(self.negatives) < self.neg_N)
        positives = np.random.choice(self.positives, self.pos_N, replace=len(self.positives) < self.pos_N)
        indices = np.concatenate([negatives, positives])
        assert len(indices) == (self.pos_N + self.neg_N)
        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.pos_N + self.neg_N


class Upsampler(Sampler):
    """
    This sampler is useful if you have a very SMALL dataset, and you don't want
    to define an epoch as a single pass through the dataset.
    """
    def __init__(self, dataset, upsample_factor):
        super().__init__(data_source=dataset)
        self.len_dataset = len(dataset)
        self.upsample_factor = upsample_factor

    def __iter__(self):
        upsampled = np.random.choice(np.arange(self.len_dataset), int(self.len_dataset * self.upsample_factor),
                                     replace=True)
        return iter(upsampled)

    def __len__(self):
        return int(self.len_dataset * self.upsample_factor)
