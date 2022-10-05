import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import List, Optional, Union

from mymi.dataset import TrainingDataset
from mymi.geometry import get_extent_centre
from mymi.regions import get_region_patch_size
from mymi.transforms import point_crop_or_pad_3D
from mymi import types

class PatchLoader:
    @staticmethod
    def build(
        datasets: Union[TrainingDataset, List[TrainingDataset]],
        region: str,
        batch_size: int = 1,
        half_precision: bool = True,
        n_folds: Optional[int] = None,
        n_samples: Optional[int] = None,
        n_workers: int = 1,
        p_foreground: float = 1,
        shuffle: bool = True,
        spacing: types.ImageSpacing3D = None,
        test_fold: Optional[int] = None,
        transform: torchio.transforms.Transform = None) -> torch.utils.data.DataLoader:
        if type(partitions) == TrainingPartition:
            partitions = [partitions]

        # Create dataset object.
        ds = LoaderDataset(partitions, region, half_precision=half_precision, n_samples=n_samples, p_foreground=p_foreground, spacing=spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=ds, num_workers=n_workers, shuffle=shuffle)

class LoaderDataset(Dataset):
    def __init__(
        self,
        datasets: List[TrainingDataset],
        region: str,
        half_precision: bool = True,
        n_samples: Optional[int] = None,
        p_foreground: float = 1,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        self._half_precision = half_precision
        self._p_foreground = p_foreground
        self._partitions = partitions
        self._patch_size = get_region_patch_size(region, spacing)
        self._region = region
        self._spacing = spacing
        self._transform = transform

        index = 0
        map_tuples = []
        for i, partition in enumerate(partitions):
            # Filter samples by requested regions.
            samples = partition.list_samples(regions=region)
            for sample in samples:
                map_tuples.append((index, (i, sample)))
                index += 1

        # Set number of samples.
        if n_samples:
            if n_samples > index:
                part_names = [p.name for p in partitions]
                raise ValueError(f"Requested '{n_samples}' samples for PatchLoader with region '{region}' from partitions '{part_names}', only '{index}' samples found.")
            else:
                self._n_samples = n_samples
        else:
            self._n_samples = index

        # Map loader indices to dataset indices.
        self._index_map = dict(map_tuples)

    def __len__(self):
        """
        returns: number of samples in the partition.
        """
        return self._n_samples

    def __getitem__(
        self,
        index: int):
        """
        returns: an (input, label) pair from the dataset.
        args:
            index: the item to return.
        """
        # Load data.
        p_idx, s_idx = self._index_map[index]
        part = self._partitions[p_idx]
        input, label = part.sample(s_idx).pair(regions=self._region)
        label = label[self._region]

        # Get description.
        desc = f'{part.dataset.name}:{part.name}:{s_idx}'

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
            subject = Subject(input=input, label=label)

            # Transform the subject.
            output = self._transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

            # Convert to numpy.
            input = input.numpy()
            label = label.numpy()

        # Roll the dice.
        if np.random.binomial(1, self._p_foreground):
            # Check that foreground voxels are present.
            if label.sum() > 0:
                input, label = self._get_foreground_patch(input, label)
            else:
                input, label = self._get_background_patch(input, label)
        else:
            input, label = self._get_background_patch(input, label)

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert dtypes
        if self._half_precision:
            input = input.astype(np.half)
        else:
            input = input.astype(np.single)
        label = label.astype(np.bool)

        return desc, input, label

    def _get_foreground_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:
        """
        returns: a patch around the OAR.
        args:
            input: the input data.
            label: the label data.
        """
        # Choose randomly from the foreground voxels.
        fg_voxels = np.argwhere(label != 0)
        fg_voxel_idx = np.random.choice(len(fg_voxels))
        centre = fg_voxels[fg_voxel_idx]

        # Extract patch around centre.
        input = point_crop_or_pad_3D(input, self._patch_size, centre, fill=input.min())        
        label = point_crop_or_pad_3D(label, self._patch_size, centre)

        return input, label

    def _get_background_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:
        # Choose a random voxel.
        centre = tuple(map(np.random.randint, self._patch_size))

        # Extract patch around centre.
        input = point_crop_or_pad_3D(input, self._patch_size, centre, fill=input.min())        
        label = point_crop_or_pad_3D(label, self._patch_size, centre)

        return input, label
