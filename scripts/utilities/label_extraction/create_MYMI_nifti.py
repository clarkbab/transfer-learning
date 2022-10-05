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
from mymi import logging

n_patients = 20
regions = ['Brain', 'Cochlea_L', 'Cochlea_R', 'OralCavity', 'Parotid_L', 'Parotid_R', 'SpinalCord', 'Submandibular_L', 'Submandibular_R']

dataset = 'HN1'
set = ds.get(dataset, 'dicom')
pats = set.list_patients()[:n_patients]

dest_dataset = 'HN1-MYMI'
nifti_set = recreate(dest_dataset)

for pat in tqdm(pats):
    patient = set.patient(pat)
    studies = patient.list_studies()
    if len(studies) == 0:
        continue
    study = patient.study(studies[0])

    # Create labels.
    for region in regions:
        if not patient.has_region(region):
            logging.info(f"Patient '{patient}' doesn't have region '{region}'.")
            continue

        # Get label.
        spacing = patient.ct_spacing()
        offset = patient.ct_offset()
        image = patient.ct_data()
        label = patient.region_data(regions=region)[region]

        # Save image.
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
        img = Nifti1Image(label.astype(np.int32), affine)
        filepath = os.path.join(nifti_set.path, 'data', region, f'{pat}.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)
