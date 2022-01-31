import math
from typing import TypeVar, Iterator, List

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)



class custom_sampler(Sampler[T_co]):
    """
    Motivated from distributed.data.sampler
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler

    An infinite sampler that provides the indexes of the dataset to be loaded by the dataloader.
    It is also able to prioritize certain samples, when requested.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int): Total number of processes using this sampler.
        rank (int): Rank of the current process.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(self, dataset: Dataset,
                 num_replicas: int,
                 rank: int,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        # dataset.file_paths is specific to vast.data_prep.ImageNetPytorch
        self.path_to_indx_mapping = dict(zip(self.dataset.file_paths,range(len(self.dataset))))
        self.pick_specific_samples = []

    def add_specific_samples(self, image_name_to_add):
        indexes_of_interest = [self.path_to_indx_mapping[name] for name in image_name_to_add]
        # Rather than extending the list of sample to process we choose to create a
        # new list every time a new batch of samples is requested.
        # This is done to avoid a possible corner case during multiprocessing.
        self.pick_specific_samples.append(indexes_of_interest)

    def __reset_iterator__(self) -> List:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        return indices

    def __subsample_from_list_by_rank__(self, list):
        return list[self.rank:self.total_size:self.num_replicas]

    def __iter__(self) -> Iterator[T_co]:
        while True:
            indices = self.__reset_iterator__()
            # subsample
            indices = self.__subsample_from_list_by_rank__(indices)
            for i in indices:
                while len(self.pick_specific_samples)>0:
                    samples_requested = self.pick_specific_samples.pop(-1)
                    yield from self.__subsample_from_list_by_rank__(samples_requested)
                yield i
            self.epoch+=1

    def __len__(self) -> int:
        return self.num_samples