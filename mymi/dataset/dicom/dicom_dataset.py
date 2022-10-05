from collections import OrderedDict
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from scipy.ndimage import center_of_mass
from skimage.draw import polygon
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union

from mymi import config
from mymi import logging
from mymi import regions
from mymi import types
from mymi.utils import append_row

from ..dataset import Dataset, DatasetType
from .dicom_patient import DICOMPatient
from .index import build_index
from .region_map import RegionMap

Z_SPACING_ROUND_DP = 2

class DICOMDataset(Dataset):
    def __init__(
        self,
        name: str):
        self._path = os.path.join(config.directories.datasets, 'dicom', name)

        # Load 'ct_from' flag.
        ct_from_name = None
        for f in os.listdir(self._path):
            match = re.match('^ct_from_(.*)$', f)
            if match:
                ct_from_name = match.group(1)

        self._ct_from = DICOMDataset(ct_from_name) if ct_from_name is not None else None
        self._global_id = f"DICOM: {name}"
        self._global_id = self._global_id + f" (CT from - {self._ct_from})" if self._ct_from is not None else self._global_id
        self._name = name
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self}' not found.")

        # Load indexes.
        filepath = os.path.join(self._path, 'index.csv')
        if not os.path.exists(filepath):
            build_index(name)
        self._index = pd.read_csv(filepath, dtype={ 'patient-id': str })
        filepath = os.path.join(self._path, 'index-errors.csv')
        self._index_errors = pd.read_csv(filepath, dtype={ 'patient-id': str })

        # Load region map.
        self.__region_map = self._load_region_map()

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def region_map(self) -> RegionMap:
        return self.__region_map

    @property
    def index_errors(self) -> pd.DataFrame:
        return self._index_errors

    def __str__(self) -> str:
        return self._global_id

    @property
    def ct_from(self) -> Optional['DICOMDataset']:
        return self._ct_from

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> DatasetType:
        return self._type

    @property
    def path(self) -> str:
        return self._path

    def to_internal(self, region: str) -> str:
        return self.__region_map.to_internal(region) if self.__region_map is not None else region

    def trimmed_errors(self) -> pd.DataFrame:
        path = os.path.join(self._path, 'hierarchy', 'trimmed', 'errors.csv')
        return pd.read_csv(path)

    def has_patient(
        self,
        id: types.PatientID) -> bool:
        """
        returns: whether the patient is present in the dataset or not.
        args:
            id: the patient ID.
        """
        return id in self.list_patients()

    def list_patients(
        self,
        regions: types.PatientRegions = 'all',
        trimmed: bool = False) -> List[str]:
        pats = list(sorted(self.index['patient-id'].unique()))

        # Filter by 'regions'.
        pats = list(filter(self._filter_patient_by_regions(regions), pats))
        return pats

    def patient(
        self,
        id: types.PatientID,
        **kwargs: Dict) -> DICOMPatient:
        return DICOMPatient(self, id, region_map=self.__region_map, **kwargs)

    def list_regions(
        self,
        n_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        trimmed: bool = False,
        use_mapping: bool = True) -> pd.DataFrame:
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients(trimmed=trimmed)

        # Filter patients.
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        pats = list(filter(self._filter_patient_by_n_pats(n_pats), pats))

        # Add patient regions.
        logging.info(f"Loading regions for dataset '{self._name}'..")
        for pat in tqdm(pats):
            try:
                pat_regions = self.patient(pat, trimmed=trimmed).list_regions(use_mapping=use_mapping)
            except ValueError as e:
                # Allow errors if we're inspecting 'trimmed' patients.
                if trimmed:
                    logging.error(e)
                else:
                    raise e

            for pat_region in pat_regions:
                data = {
                    'patient-id': pat,
                    'region': pat_region
                }
                df = append_row(df, data)

        # Set column types.
        df = df.astype(cols)

        return df

    def _load_index(self) -> pd.DataFrame:
        filepath = os.path.join(self._path, 'index.csv')
        index = pd.read_csv(filepath)
        return index

    def _load_region_map(self) -> Optional[RegionMap]:
        # Check for region map.
        filepath = os.path.join(self._path, 'region-map.csv')
        if os.path.exists(filepath):
            # Load map file.
            map_df = pd.read_csv(filepath)

            # Check that internal region names are entered correctly.
            for n in map_df.internal:
                if not regions.is_region(n):
                    raise ValueError(f"Error in region map for dataset '{self._name}', '{n}' is not an internal region.")
            
            return RegionMap(map_df)
        else:
            return None

    def _filter_patient_by_n_pats(
        self,
        n_pats: int) -> Callable[[str], bool]:
        def fn(id):
            if n_pats == 'all' or fn.n_included < n_pats:
                fn.n_included += 1
                return True
            else:
                return False

        # Assign state to the function.
        fn.n_included = 0
        return fn

    def _filter_patient_by_pat_ids(
        self,
        pat_ids: Union[str, List[str]]) -> Callable[[str], bool]:
        def fn(id):
            if ((isinstance(pat_ids, str) and (pat_ids == 'all' or id == pat_ids)) or
                ((isinstance(pat_ids, list) or isinstance(pat_ids, np.ndarray) or isinstance(pat_ids, tuple)) and id in pat_ids)):
                return True
            else:
                return False
        return fn

    def _filter_patient_by_regions(
        self,
        regions: types.PatientRegions,
        use_mapping: bool = True) -> Callable[[str], bool]:
        def fn(id):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.patient(id).has_region(regions, use_mapping=use_mapping)
            else:
                pat_regions = self.patient(id).list_regions(use_mapping=use_mapping)
                if len(np.intersect1d(regions, pat_regions)) != 0:
                    return True
                else:
                    return False
        return fn
