from typing import Optional

from mymi.dataset.dicom import DICOMDataset
from mymi import types

from ..processing import convert_to_nifti as ctn
# from ..processing import convert_to_training as ctt

def convert_to_nifti(
    dataset: str,
    regions: types.PatientRegions = 'all',
    anonymise: bool = False) -> None:
    set = DICOMDataset(dataset)
    ctn(set, regions=regions, anonymise=anonymise)

# def convert_to_training(
#     dataset: str,
#     dest_dataset: str,
#     dilate_regions: Optional[types.PatientRegions] = None,
#     p_test: float = 0.2,
#     p_train: float = 0.6,
#     p_val: float = 0.2,
#     random_seed: int = 42,
#     regions: types.PatientRegions = 'all',
#     size: Optional[types.ImageSize3D] = None,
#     spacing: Optional[types.ImageSpacing3D] = None,
#     use_mapping: bool = True):
#     set = DICOMDataset(dataset)
#     ctt(set, dest_dataset, dilate_regions=dilate_regions, p_test=p_test, p_train=p_train, p_val=p_val, 
#         random_seed=random_seed, regions=regions, size=size, spacing=spacing, use_mapping=use_mapping)
