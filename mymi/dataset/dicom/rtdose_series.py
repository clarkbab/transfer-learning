import numpy as np
import pandas as pd
import pydicom as dcm
from mymi import types

from .dicom_series import DICOMModality, DICOMSeries
from mymi.transforms import resample_3D

class RTDOSESeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str) -> None:
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id

        # Get index.
        index = self._study.index
        index = index[(index.modality == 'RTDOSE') & (index['series-id'] == id)]
        self._index = index
        self._path = index.iloc[0].filepath

        # Check that series exists.
        if len(index) == 0:
            raise ValueError(f"RTDOSE series '{self}' not found in index for study '{study}'.")

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTDOSE

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def study(self) -> str:
        return self._study

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return self._global_id

    def get_rtdose(self) -> dcm.dataset.FileDataset:
        filepath = self._index.iloc[0].filepath
        rtdose = dcm.read_file(filepath)
        return rtdose

    @property
    def data(self) -> np.ndarray:
        patient = self.study.patient
        rtdose = self.get_rtdose()
        data = np.transpose(rtdose.pixel_array)
        data = rtdose.DoseGridScaling * data
        data = resample_3D(data, origin=self.offset, spacing=self.spacing, output_origin=patient.ct_offset, output_size=patient.ct_size, output_spacing=patient.ct_spacing) 
        return data

    @property
    def offset(self) -> types.PhysPoint3D:
        rtdose = self.get_rtdose()
        offset = rtdose.ImagePositionPatient
        offset = tuple(int(s) for s in offset)
        return offset

    @property
    def size(self) -> types.ImageSize3D:
        return self.data.shape

    @property
    def spacing(self) -> types.ImageSpacing3D:
        rtdose = self.get_rtdose()
        spacing_x_y = rtdose.PixelSpacing 
        z_diffs = np.unique(np.diff(rtdose.GridFrameOffsetVector))
        assert len(z_diffs) == 1
        spacing_z = z_diffs[0]
        spacing = tuple(np.append(spacing_x_y, spacing_z))
        return spacing
