from numpy.lib.arraysetops import intersect1d
from mymi.types.types import PatientRegions
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Union

from mymi.regions import to_list
from mymi import types

class TrainingSample:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        id: Union[str, int]):
        self.__dataset = dataset
        self.__id = str(id)
        self.__global_id = f'{self.__dataset} - {self.__id}'
        self.__spacing = self.__dataset.params['spacing']

        # Load sample index.
        if self.__id not in self.__dataset.list_samples():
            raise ValueError(f"Sample '{self.__id}' not found for dataset '{self.__dataset}'.")
        index = self.__dataset.index
        index = index[index['sample-id'] == self.__id]
        self._index = index

    @property
    def index(self) -> str:
        return self._index

    @property
    def description(self) -> str:
        return self.__global_id

    def __str__(self) -> str:
        return self.__global_id

    @property
    def id(self) -> str:
        return self.__id

    @property
    def spacing(self) -> types.ImageSpacing3D:
        return self.__spacing

    def list_regions(self) -> List[str]:
        return list(sorted(self._index.region))

    def has_region(
        self,
        region: str) -> bool:
        return region in self.list_regions()

    @property
    def origin(self) -> Tuple:
        idx = self.__dataset.index
        record = idx[idx['sample-id'] == self.__id].iloc[0]
        return (record['origin-dataset'], record['patient-id'])

    @property
    def input(self) -> np.ndarray:
        # Load the input data.
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}.npz')
        data = np.load(filepath)['data']
        return data

    def label(
        self,
        regions: types.PatientRegions = 'all') -> Dict[str, np.ndarray]:
        # Convert regions to list.
        if type(regions) == str:
            if regions == 'all':
                regions = list(sorted(self.list_regions()))
            else:
                regions = [regions]

        # Load the label data.
        data = {}
        for region in regions:
            filepath = os.path.join(self.__dataset.path, 'data', 'labels', region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Region '{region}' not found for sample '{self}'.")
            label = np.load(filepath)['data']
            data[region] = label

        return data

    def pair(
        self,
        regions: types.PatientRegions = 'all') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.input, self.label(regions)
