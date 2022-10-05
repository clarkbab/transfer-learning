import os
import pandas as pd
from typing import Dict, List, Optional

from .ct_series import CTSeries
from .dicom_series import DICOMSeries
from .region_map import RegionMap
from .rtdose_series import RTDOSESeries
from .rtplan_series import RTPLANSeries
from .rtstruct_series import RTSTRUCTSeries

class DICOMStudy:
    def __init__(
        self,
        patient: 'DICOMPatient',
        id: str,
        region_map: Optional[RegionMap] = None):
        self._patient = patient
        self._id = id
        self._region_map = region_map
        self._global_id = f"{patient} - {id}"

        # Get study index.
        index = self._patient.index
        index = index[index['study-id'] == id]
        self._index = index 
    
        # Check that study ID exists.
        if len(index) == 0:
            raise ValueError(f"Study '{self}' not found in index for patient '{patient}'.")

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    def __str__(self) -> str:
        return self._global_id

    @property
    def patient(self) -> str:
        return self._patient

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def list_series(
        self,
        modality: str) -> List[str]:
        index = self._index
        index = index[index.modality == modality]
        series = list(sorted(index['series-id'].unique()))
        return series

    def series(
        self,
        id: str,
        modality: str,
        **kwargs: Dict) -> DICOMSeries:
        if modality == 'CT':
            return CTSeries(self, id, **kwargs)
        elif modality == 'RTSTRUCT':
            return RTSTRUCTSeries(self, id, region_map=self._region_map, **kwargs)
        elif modality == 'RTPLAN':
            return RTPLANSeries(self, id, **kwargs)
        elif modality == 'RTDOSE':
            return RTDOSESeries(self, id, **kwargs)
        else:
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")
