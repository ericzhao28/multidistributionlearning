import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        self.group_dict = {}

        if (self.dataset.y_array is not None) and (self.dataset.group_array is not None):
            y_array = self.dataset.y_array
            group_array = self.dataset.group_array
        else:
            group_array = np.empty(len(self))
            y_array = np.empty(len(self))
            for i,(x,y,g) in enumerate(self):
                group_array[i] = g
                y_array[i] = y


        for i, g in enumerate(group_array):
            if g not in self.group_dict:
                self.group_dict[g] = []
            self.group_dict[g].append(i)

        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g in self:
            return x.size()

    def get_loader(self, train, reweight_groups, epoch_size=None, replacement=True, partitioner=False, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        elif partitioner:
            group_weights = len(self) / self._group_counts
            weights = self.get_partitioner(group_weights)
            sampler = WeightedRandomSampler(weights, epoch_size, replacement=replacement)
            shuffle = False

        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self) / self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            if not epoch_size:
                epoch_size = len(self)
            sampler = WeightedRandomSampler(weights, epoch_size, replacement=replacement)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader

    def get_partitioner(self, group_weights):
        weights = torch.zeros_like(self._group_array)

        for g, indices in self.group_dict.items():
            count = min([int(group_weights[g] * len(self)), len(indices)])
            good_indices = np.random.choice(indices, size=(count,), replace=False)
            weights[good_indices] = 1

        return weights + 1e-5
