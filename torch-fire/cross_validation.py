from typing import Callable

import torch

from torch.utils.data import Dataset, DataLoader, Subset


def cross_validation(n_folds: int, dataset: Dataset, func: Callable, *args, **kwargs):
    for i in range(n_folds):


