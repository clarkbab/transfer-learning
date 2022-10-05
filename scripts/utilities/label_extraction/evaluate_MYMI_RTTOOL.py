import math
import os
from os.path import dirname as up
import pandas as pd
import pathlib
import sys
from tqdm import tqdm

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config
from mymi import dataset
from mymi import logging
from mymi.metrics import dice, distances

# Get datasets.
mymi_name = 'MYMI'
rttool_name = 'RTTOOL'
mymi_set = dataset.get(f'HN1-{mymi_name}', 'nifti')
rttool_set = dataset.get(f'HN1-{rttool_name}', 'nifti')

# Get patients.
mymi_pats = mymi_set.list_patients()
pats = rttool_set.list_patients()

cols = {
    'patient-id': str,
    'region': str,
    'metric': str,
    'value': float
}
df = pd.DataFrame(columns=cols.keys())

for pat in tqdm(pats):
    if pat not in mymi_pats:
        logging.error(f"Patient '{pat}' not found in MYMI set.")
        continue    

    # Get spacing.
    mymi_patient = mymi_set.patient(pat)
    mymi_spacing = mymi_patient.ct_spacing()
    patient = rttool_set.patient(pat)
    spacing = patient.ct_spacing()
    for i, (mymi_spacing_i, spacing_i) in enumerate(zip(mymi_spacing, spacing)):
        assert math.isclose(mymi_spacing_i, spacing_i, abs_tol=1e-6), f"Spacing not equal for axis '{i}', got mymi '{mymi_spacing_i}' and DicomRTTool '{spacing_i}'."

    # Get regions.
    mymi_regions = mymi_patient.list_regions()
    regions = patient.list_regions()
    
    # Evaluate labels.
    for region in regions:
        if region not in mymi_regions:
            logging.error(f"Region '{region}' not found for patient '{pat}' MYMI set.")
            continue    
        
        data = {
            'patient-id': pat,
            'region': region,
        }

        # Get labels.
        mymi_label = mymi_patient.region_data(regions=region)[region]
        label = patient.region_data(regions=region)[region]
        
        # Get DSC.
        data['metric'] = 'dice'
        data['value'] = dice(mymi_label, label)
        df = df.append(data, ignore_index=True)
        
        # Distances.
        dists = distances(mymi_label, label, spacing)
        for metric, value in dists.items():
            data['metric'] = metric
            data['value'] = value
            df = df.append(data, ignore_index=True)

# Set types.
df = df.astype(cols)

# Save evaluation.
filepath = os.path.join(config.directories.files, f'{mymi_name}-{rttool_name}-evaluation.csv')
os.makedirs(os.path.dirname(filepath), exist_ok=True)
df.to_csv(filepath, index=False)
