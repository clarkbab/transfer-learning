from .dicom_dataset import DicomDataset
from .roi_data import ROIData
from .rtstruct_converter import RTSTRUCTConverter

import os
import shutil
from typing import List

from mymi import config

from .dicom_dataset import DicomDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'dicom')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    os.makedirs(ds_path)
    return NiftiDataset(name)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    destroy(name)
    return create(name)
