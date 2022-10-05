import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional, Union

from mymi import config
from mymi import types
from mymi.utils import append_row, load_csv

from ..dataset import Dataset, DatasetType
from .nifti_patient import NIFTIPatient

class NIFTIDataset(Dataset):
    def __init__(
        self,
        name: str):
        self._global_id = f"NIFTI: {name}"
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'nifti', name)
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self}' not found.")
    
    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def path(self) -> str:
        return self._path

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    @property
    def anon_manifest(self) -> Optional[pd.DataFrame]:
        man_df = load_csv('anon-maps', f'{self._name}.csv')
        man_df = man_df.astype({ 'anon-id': str, 'patient-id': str })
        return man_df

    def list_patients(
        self,
        regions: types.PatientRegions = 'all') -> List[str]:
        """
        returns: a list of NIFTI IDs.
        """
        # Load patients.
        ct_path = os.path.join(self._path, 'data', 'ct')
        files = list(sorted(os.listdir(ct_path)))
        pats = [f.replace('.nii.gz', '') for f in files]

        # Filter by 'regions'.
        pats = list(filter(self._filter_patient_by_regions(regions), pats))
        return pats

    def list_regions(self) -> pd.DataFrame:
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pat_ids = self.list_patients()

        # Add patient regions.
        for pat_id in pat_ids:
            for region in self.patient(pat_id).list_regions():
                data = {
                    'patient-id': pat_id,
                    'region': region,
                }
                df = append_row(df, data)

        # Set column types.
        df = df.astype(cols)

        return df

    def patient(
        self,
        id: Union[int, str],
        by_dicom_id: bool = False) -> NIFTIPatient:
        if by_dicom_id:
            man_df = self.anon_manifest
            man_df = man_df[man_df['patient-id'] == str(id)]
            id = man_df.iloc[0]['anon-id']
        return NIFTIPatient(self, id)

    def _filter_patient_by_regions(
        self,
        regions: types.PatientRegions) -> Callable[[str], bool]:
        def func(id):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.patient(id).has_region(regions)
            else:
                pat_regions = self.patient(id).list_regions()
                if len(np.intersect1d(regions, pat_regions)) != 0:
                    return True
                else:
                    return False
        return func
