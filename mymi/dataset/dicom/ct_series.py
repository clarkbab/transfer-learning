import math
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import List

from mymi import cache
from mymi import config
from mymi import types

from .dicom_series import DICOMModality, DICOMSeries

CLOSENESS_ABS_TOL = 1e-10;

class CTSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str):
        self.__global_id = f"{study} - {id}"
        self.__study = study
        self.__id = id

        # Load index.
        index = self.__study.index
        index = index[(index.modality == 'CT') & (index['series-id'] == id)]
        self.__index = index
        self.__paths = list(self.__index['filepath'])
        
        # Check that series exists.
        if len(index) == 0:
            raise ValueError(f"CT series '{self}' not found in index for study '{study}'.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def id(self) -> str:
        return self.__id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.CT

    @property
    def paths(self) -> str:
        return self.__paths

    @property
    def study(self) -> str:
        return self.__study
    
    def __str__(self) -> str:
        return self.__global_id

    def get_cts(self) -> List[dcm.dataset.FileDataset]:
        ct_paths = list(self.__index['filepath'])
        cts = [dcm.read_file(f) for f in ct_paths]
        cts = list(sorted(cts, key=lambda c: c.ImagePositionPatient[2]))
        return cts

    def get_first_ct(self) -> dcm.dataset.FileDataset:
        ct_paths = list(self.__index['filepath'])
        ct = dcm.read_file(ct_paths[0])
        return ct

    @property
    def offset(self) -> types.PhysPoint3D:
        cts = self.get_cts()
        offset = cts[0].ImagePositionPatient
        offset = tuple(int(s) for s in offset)
        return offset

    def orientation(self) -> types.ImageSpacing3D:
        cts = self.get_cts()

        # Get the orientation.
        orientation = (
            (
                cts[0].ImageOrientationPatient[0],
                cts[0].ImageOrientationPatient[1],
                cts[0].ImageOrientationPatient[2]
            ),
            (
                cts[0].ImageOrientationPatient[3],
                cts[0].ImageOrientationPatient[4],
                cts[0].ImageOrientationPatient[5]
            )
        )
        return orientation

    @property
    def size(self) -> types.ImageSpacing3D:
        cts = self.get_cts()

        # Get size - relies on hierarchy filtering (i.e. removing patients with missing slices).
        size = (
            cts[0].pixel_array.shape[1],
            cts[0].pixel_array.shape[0],
            len(cts)
        )
        return size

    @property
    def spacing(self) -> types.ImageSpacing3D:
        cts = self.get_cts()

        # Get spacing - relies on consistent spacing checks during index building.
        spacing = (
            float(cts[0].PixelSpacing[0]),
            float(cts[0].PixelSpacing[1]),
            np.abs(cts[1].ImagePositionPatient[2] - cts[0].ImagePositionPatient[2])
        )
        return spacing

    @property
    def data(self) -> np.ndarray:
        # Load series CT dicoms.
        cts = self.get_cts()

        # Load CT summary info.
        size = self.size
        offset = self.offset
        spacing = self.spacing
        
        # Create CT data array.
        data = np.zeros(shape=size)
        for ct in cts:
            # Convert to HU. Transpose to (x, y) coordinates, 'pixel_array' returns
            # row-first image data.
            ct_data = np.transpose(ct.pixel_array)
            ct_data = ct.RescaleSlope * ct_data + ct.RescaleIntercept

            # Get z index.
            z_offset =  ct.ImagePositionPatient[2] - offset[2]
            z_idx = int(round(z_offset / spacing[2]))

            # Add data.
            data[:, :, z_idx] = ct_data

        return data
