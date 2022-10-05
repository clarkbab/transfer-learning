import os
import pandas as pd
import pydicom as dcm

from .rtstruct_series import RTSTRUCTSeries
from .dicom_series import DICOMModality, DICOMSeries

class RTPLANSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str) -> None:
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id

        # Get index.
        index = self._study.index
        index = index[(index.modality == 'RTPLAN') & (index['series-id'] == id)]
        self._index = index
        if len(index) != 1:
            raise ValueError(f"Index length '{len(index)}' for  series '{self}'.")
        self._path = index.iloc[0]['filepath']

        # Check that series exists.
        if len(index) == 0:
            raise ValueError(f"RTPLAN series '{self}' not found in index for study '{study}'.")

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def id(self) -> str:
        return self._id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTPLAN

    @property
    def path(self) -> str:
        return self._path

    @property
    def study(self) -> str:
        return self._study

    def __str__(self) -> str:
        return self._global_id

    def get_rtplan(self) -> dcm.dataset.FileDataset:
        filepath = self._index.iloc[0].filepath
        rtplan = dcm.read_file(filepath)
        return rtplan
