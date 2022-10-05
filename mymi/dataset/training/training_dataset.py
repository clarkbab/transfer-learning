import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional, Union

from mymi import config
from mymi import types

from ..dataset import Dataset, DatasetType
from .training_sample import TrainingSample

class TrainingDataset(Dataset):
    def __init__(
        self,
        name: str,
        check_processed: bool = True,
        load_index: bool = True):
        self.__name = name
        self.__global_id = f"TRAINING: {self.__name}"
        self.__path = os.path.join(config.directories.datasets, 'training', self.__name)

        # Check if dataset exists.
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

        # Check if processing from NIFTI has completed.
        if check_processed:
            path = os.path.join(self.__path, '__CONVERT_FROM_NIFTI_START__')
            if os.path.exists(path):
                path = os.path.join(self.__path, '__CONVERT_FROM_NIFTI_END__')
                if not os.path.exists(path):
                    raise ValueError(f"Dataset '{self}' processing from NIFTI not completed. To override check use 'check_processed=False'.")

        # Load data index.
        if load_index:
            filepath = os.path.join(self.__path, 'index.csv')
            self.__index = pd.read_csv(filepath).astype({ 'patient-id': str, 'sample-id': str })

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def description(self) -> str:
        return self.__global_id

    def __str__(self) -> str:
        return self.__global_id

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    @property
    def params(self) -> pd.DataFrame:
        filepath = os.path.join(self.__path, 'params.csv')
        df = pd.read_csv(filepath)
        params = df.iloc[0].to_dict()
        
        # Replace special columns.
        cols = ['size', 'spacing']
        for col in cols:
            if col == 'None':
                params[col] = None
            else:
                params[col] = eval(params[col])
        return params

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def patient_id(
        self,
        sample_idx: int) -> types.PatientID:
        df = self.__index[self.__index['sample-id'] == sample_idx]
        if len(df) == 0:
            raise ValueError(f"Sample '{sample_idx}' not found for dataset '{self}'.")
        pat_id = df['patient-id'].iloc[0] 
        return pat_id

    def list_samples(
        self,
        regions: Optional[Union[str, List[str]]] = None) -> List[int]:
        if type(regions) == str:
            regions = [regions]

        # Filter by regions.
        if regions is not None:
            index = self.__index[self.__index.region.isin(regions)]
        else:
            index = self.__index

        # Get sample IDs.
        sample_ids = list(sorted(index['sample-id'].unique()))

        return sample_ids

    def sample(
        self,
        sample_id: Union[int, str],
        by_patient_id: bool = False) -> TrainingSample:
        # Look up sample by patient ID.
        if by_patient_id:
            sample_id = self.__index[self.__index['patient-id'] == sample_id].iloc[0]['sample-id']

        return TrainingSample(self, sample_id)
