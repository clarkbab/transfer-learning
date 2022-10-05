import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Callable, List, Optional, Tuple, Union

from mymi import types
from mymi.dataset import get as get_ds
from mymi.dataset.training import TrainingDataset

class PatchLoader:
    @staticmethod
    def build_loaders(
        datasets: Union[TrainingDataset, List[TrainingDataset]],
        region: str,
        batch_size: int = 1,
        half_precision: bool = True,
        n_folds: Optional[int] = None, 
        n_train: Optional[int] = None,
        n_workers: int = 1,
        random_seed: int = 42,
        spacing: types.ImageSpacing3D = None,
        test_fold: Optional[int] = None,
        transform: torchio.transforms.Transform = None,
        p_val: float = .2) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        if type(datasets) == TrainingDataset:
            datasets = [datasets]
        if n_folds and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")

        # Get all samples.
        all_samples = []
        for ds_i, dataset in enumerate(datasets):
            samples = dataset.list_samples(regions=region)
            for s_i in samples:
                all_samples.append((ds_i, s_i))

        # Shuffle samples.
        np.random.seed(random_seed)
        np.random.shuffle(all_samples)

        # Split samples into folds.
        if n_folds:
            n_samples = len(all_samples)
            len_fold = int(np.floor(n_samples / n_folds))
            folds = []
            for i in range(n_folds):
                fold = all_samples[i * len_fold:(i + 1) * len_fold]
                folds.append(fold)

            # Determine train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for n_folds=5). This ensures that when we 
            # take a subset of samples (n_train != None), we get different training samples
            # for each of the k-folds.
            train_folds = list((np.array(range(n_folds)) + (test_fold + 1)) % 5)
            train_folds.remove(test_fold)

            # Get train and test data.
            train_samples = []
            for i in train_folds:
                train_samples += folds[i]
            test_samples = folds[test_fold] 
        else:
            train_samples = all_samples

        # Take subset of train samples.
        if n_train is not None:
            train_samples = train_samples[:n_train]

        # Split train into NN train and validation data.
        n_nn_train = int(len(train_samples) * (1 - p_val))
        nn_train_samples = train_samples[:n_nn_train]
        nn_val_samples = train_samples[n_nn_train:] 

        # Create train loader.
        train_ds = TrainingDataset(datasets, region, nn_train_samples, half_precision=half_precision, spacing=spacing, transform=transform)
        train_loader = DataLoader(batch_size=batch_size, dataset=train_ds, num_workers=n_workers, shuffle=True)

        # Create validation loader.
        val_ds = TrainingDataset(datasets, region, nn_val_samples, half_precision=half_precision)
        val_loader = DataLoader(batch_size=batch_size, dataset=val_ds, num_workers=n_workers, shuffle=False)

        # Create test loader.
        if n_folds:
            test_ds = TestDataset(datasets, test_samples) 
            test_loader = DataLoader(batch_size=batch_size, dataset=test_ds, num_workers=n_workers, shuffle=False)
            return train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader

class TrainingDataset(Dataset):
    def __init__(
        self,
        datasets: List[TrainingDataset],
        region: str,
        samples: List[Tuple[int, int]],
        half_precision: bool = True,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        self._datasets = datasets
        self._half_precision = half_precision
        self._region = region
        self._spacing = spacing
        self._transform = transform
        if transform:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Record number of samples.
        self._n_samples = len(samples)

        # Map loader indices to dataset indices.
        self._sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

    def __len__(self):
        return self._n_samples

    def __getitem__(
        self,
        index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Load data.
        ds_i, s_i = self._sample_map[index]
        dataset = self._datasets[ds_i]
        input, labels = dataset.sample(s_i).pair(regions=self._region)
        label = labels[self._region]

        # Get description.
        desc = f'{dataset.name}:{s_i}'

        # Perform transform.
        if self._transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self._spacing[0], 0, 0, 0],
                [0, self._spacing[1], 0, 0],
                [0, 0, self._spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject_kwargs = { 'input': input }
            for r, d in label.items():
                subject_kwargs[r] = d
            subject = Subject({
                'input': input,
                'label': label
            })

            # Transform the subject.
            output = self._transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

            # Convert to numpy.
            input = input.numpy()
            label = label.numpy()

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert dtypes.
        if self._half_precision:
            input = input.astype(np.half)
        else:
            input = input.astype(np.single)
        label = label.astype(bool)

        return desc, input, label
    
class TestDataset(Dataset):
    def __init__(
        self,
        datasets: List[TrainingDataset],
        samples: List[Tuple[int, int]]):
        self._datasets = datasets

        # Record number of samples.
        self._n_samples = len(samples)

        # Map loader indices to dataset indices.
        self._sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

    def __len__(self):
        return self._n_samples

    def __getitem__(
        self,
        index: int) -> Tuple[str, str]:
        # Load data.
        ds_i, s_i = self._sample_map[index]
        set = self._datasets[ds_i]
        return set.sample(s_i).origin
