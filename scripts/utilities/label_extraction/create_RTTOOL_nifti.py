from DicomRTTool.ReaderWriter import DicomReaderWriter
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
from os.path import dirname as up
import pathlib
import sys
from tqdm import tqdm

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import dataset as ds
from mymi.dataset.nifti import recreate

n_patients = 20
hn1_regions = ['Brain', 'Cochlea-Left', 'Cochlea-Right', 'Oral-Cavity', 'Parotid-Left', 'Parotid-Right', 'Spinal-Cord', 'Submandibular-Gland-Left', 'Submandibular-Gland-Right']
regions = ['Brain', 'Cochlea_L', 'Cochlea_R', 'OralCavity', 'Parotid_L', 'Parotid_R', 'SpinalCord', 'Submandibular_L', 'Submandibular_R']
assert len(hn1_regions) == len(regions)

dataset = 'HN1'
set = ds.get(dataset, 'dicom')
pats = set.list_patients()[:n_patients]

dest_dataset = 'HN1-RTTOOL'
nifti_set = recreate(dest_dataset)

for pat in tqdm(pats):
    patient = set.patient(pat)
    studies = patient.list_studies()
    if len(studies) == 0:
        continue
    study = patient.study(studies[0])

    # Create masks.
    reader = DicomReaderWriter()
    reader.walk_through_folders(study.path)
    pat_regions = reader.return_rois(print_rois=False)
    for hn1_region, region in zip(hn1_regions, regions):
        if hn1_region.lower() not in pat_regions:
            print(f"Patient '{pat}' doesn't have region '{hn1_region}'.")
            continue

        # Get label.
        reader.set_contour_names_and_associations(Contour_Names=[hn1_region]) 
        reader.get_images_and_mask()
        image = np.moveaxis(reader.ArrayDicom, (0, 1, 2), (2, 1, 0))
        label = np.moveaxis(reader.mask, (0, 1, 2), (2, 1, 0))

        # Save image.
        spacing = reader.dicom_handle.GetSpacing()
        offset = reader.dicom_handle.GetOrigin()
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(image, affine) 
        filepath = os.path.join(nifti_set.path, 'data', 'ct', f'{pat}.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Save label.
        img = Nifti1Image(label, affine)
        filepath = os.path.join(nifti_set.path, 'data', region, f'{pat}.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)
