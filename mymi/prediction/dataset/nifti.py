from mymi.reporting.loaders import load_loader_manifest
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, Union

from ..prediction import get_localiser_prediction
from mymi import config
from mymi import dataset as ds
from mymi.geometry import get_box, get_extent, get_extent_centre, get_extent_width_mm
from mymi import logging as log
from mymi.loaders import Loader
from mymi.models import replace_checkpoint_alias
from mymi.models.systems import Localiser, Segmenter
from mymi.transforms import crop_foreground_3D
from mymi.regions import RegionNames, get_region_patch_size
from mymi.transforms import top_crop_or_pad_3D, crop_or_pad_3D, resample_3D
from mymi import types
from mymi.utils import append_row, load_csv

def get_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.Model,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
    # Load data.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device)

    return pred

def create_patient_localiser_prediction(
    datasets: Union[str, List[str]],
    pat_ids: Union[str, List[str]],
    localiser: Union[types.ModelName, types.Model],
    device: Optional[torch.device] = None,
    logging: bool = True,
    savepath: Optional[str] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    if type(pat_ids) == str:
        pat_ids = [pat_ids]
    if len(datasets) == 1 and len(pat_ids) != 1:
        # Broadcast datasets.
        datasets = datasets * len(pat_ids)
    assert len(datasets) == len(pat_ids)

    # Load localiser.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            log.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            log.info('Predicting on CPU...')

    for dataset, pat_id in zip(datasets, pat_ids):
        if logging:
            log.info(f"Creating prediction for patient '({dataset}, {pat_id})', localiser '{localiser.name}'.")

        # Load dataset.
        set = ds.get(dataset, 'nifti')

        # Make prediction.
        pred = get_patient_localiser_prediction(dataset, pat_id, localiser, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(set.path, 'predictions', 'localiser', *localiser.name, f'{pat_id}.npz') 
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_localiser_predictions_for_first_n_pats(
    n_pats: int,
    region: str,
    localiser: types.ModelName,
    savepath: Optional[str] = None) -> None:
    localiser = Localiser.load(*localiser)
    log.info(f"Making localiser predictions for NIFTI datasets for region '{region}', first '{n_pats}' patients in 'all-patients.csv'.")

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        log.info('Predicting on CPU...')

    # Get dataset/patient IDs.
    create_patient_localiser_prediction(*df, localiser, device=device, logging=False, savepath=savepath)

def create_localiser_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    log.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        log.info('Predicting on CPU...')

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            create_patient_localiser_prediction(dataset, pat_id, localiser, device=device, logging=False)

def load_patient_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    raise_error: bool = True) -> Optional[np.ndarray]:
    localiser = replace_checkpoint_alias(*localiser)

    # Load prediction.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', localiser '{localiser}'.")
        else:
            return None
    pred = np.load(filepath)['data']

    return pred

def load_patient_localiser_centre(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    raise_error: bool = True) -> types.Point3D:
    seg = load_patient_localiser_prediction(dataset, pat_id, localiser, raise_error=raise_error)
    if not raise_error and seg is None:
        return None
    ext_centre = get_extent_centre(seg)
    return ext_centre

def get_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    region: str,
    loc_centre: types.Point3D,
    segmenter: Union[types.Model, types.ModelName],
    probs: bool = False,
    seg_spacing: types.ImageSpacing3D = (1, 1, 2),
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    # Load model.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)
    segmenter.eval()
    segmenter.to(device)

    # Load patient CT data and spacing.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample_3D(input, spacing=spacing, output_spacing=seg_spacing) 

    # Get localiser centre on downsampled image.
    scaling = np.array(spacing) / seg_spacing
    loc_centre = tuple(int(el) for el in scaling * loc_centre)

    # Extract segmentation patch.
    resampled_size = input.shape
    patch_size = get_region_patch_size(region, seg_spacing)
    patch = get_box(loc_centre, patch_size)
    input = crop_or_pad_3D(input, patch, fill=input.min())

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = segmenter(input, probs=probs)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Crop/pad to the resampled size, i.e. before patch extraction.
    rev_patch_min, rev_patch_max = patch
    rev_patch_min = tuple(-np.array(rev_patch_min))
    rev_patch_max = tuple(np.array(rev_patch_min) + resampled_size)
    rev_patch_box = (rev_patch_min, rev_patch_max)
    pred = crop_or_pad_3D(pred, rev_patch_box)

    # Resample to original spacing.
    pred = resample_3D(pred, spacing=seg_spacing, output_spacing=spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    return pred

def create_patient_segmenter_prediction(
    datasets: Union[str, List[str]],
    pat_ids: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: Union[types.Model, types.ModelName],
    device: Optional[torch.device] = None,
    logging: bool = True,
    probs: bool = False,
    raise_error: bool = False,
    savepath: Optional[str] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    if type(pat_ids) == str:
        pat_ids = [pat_ids]
    if len(datasets) == 1 and len(pat_ids) != 1:
        # Broadcast datasets.
        datasets = datasets * len(pat_ids)
    assert len(datasets) == len(pat_ids)
    localiser = replace_checkpoint_alias(*localiser)

    # Load segmenter.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            log.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            log.info('Predicting on CPU...')

    for dataset, pat_id in zip(datasets, pat_ids):
        if logging:
            log.info(f"Creating prediction for patient '({dataset}, {pat_id})', localiser '{localiser.name}'.")

        # Get segmenter prediction.
        loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
        if loc_centre is None:
            # Create empty pred.
            if raise_error:
                raise ValueError(f"No 'loc_centre' returned from localiser.")
            else:
                ct_data = set.patient(pat_id).ct_data
                pred = np.zeros_like(ct_data, dtype=bool) 
        else:
            pred = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, device=device)

        # Save segmentation.
        if probs:
            filename = f'{pat_id}-prob.npz'
        else:
            filename = f'{pat_id}.npz'
        if savepath is None:
            savepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter.name, filename) 
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_segmenter_predictions_from_csv(
    n_pats: int,
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    savepath: Optional[str] = None) -> None:
    log.info(f"Making segmenter predictions for NIFTI datasets for region '{region}', first '{n_pats}' patients in 'all-patients.csv'.")

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        log.info('Predicting on CPU...')

    for _, (dataset, pat_id) in df.iterrows():
        # Get segmenter that wasn't trained using this patient.
        create_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter, device=device, logging=False, savepath=savepath)

def create_segmenter_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.load(*segmenter)
    log.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        log.info('Predicting on CPU...')

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            create_patient_segmenter_prediction(dataset, pat_id, region, localiser, segmenter, device=device, logging=False)

def load_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    raise_error: bool = True,
    use_model_manifest: bool = False) -> Optional[np.ndarray]:
    localiser = replace_checkpoint_alias(*localiser, use_manifest=use_model_manifest)
    segmenter = replace_checkpoint_alias(*segmenter, use_manifest=use_model_manifest)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    if config.environ('PETER_MAC_HACK') == 'True':
        base_path = 'S:\\ImageStore\\HN_AI_Contourer\\short\\nifti'
        if dataset == 'PMCC-HN-TEST':
            pred_path = os.path.join(base_path, 'test')
        elif dataset == 'PMCC-HN-TRAIN':
            pred_path = os.path.join(base_path, 'train')
    else:
        pred_path = os.path.join(set.path, 'predictions')
    filepath = os.path.join(pred_path, 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'. Path: {filepath}")
        else:
            return None
    npz_file = np.load(filepath)
    seg = npz_file['data']
    
    return seg

def save_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    data: np.ndarray) -> None:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    np.savez_compressed(filepath, data=data)

def create_two_stage_predictions_for_first_n_pats(n_pats: int) -> None:
    datasets = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
    log.info(f"Making segmenter predictions for NIFTI datasets for all regions for first '{n_pats}' patients in 'all-patients.csv'.")

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')
    df = df.astype({ 'patient-id': str })
    df = df.head(n_pats)

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        log.info('Predicting on CPU...')

    cols = {
        'region': str,
        'model': str
    }

    for _, (dataset, pat_id) in tqdm(df.iterrows()):
        index_df = pd.DataFrame(columns=cols.keys())

        for region in RegionNames:
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')

            # Find fold for which this dataset/pat_id was in the 'test' loader.
            for test_fold in range(5):
                man_df = load_loader_manifest(datasets, region, test_fold=test_fold)
                man_df = man_df[(man_df.loader == 'test') & (man_df['origin-dataset'] == dataset) & (man_df['origin-patient-id'] == pat_id)]
                if len(man_df) == 1:
                    break
            
            # Select segmenter that didn't include this patient for training.
            if len(man_df) != 0:
                # Patient was excluded when training model for 'test_fold'.
                segmenter = (f'segmenter-{region}', f'clinical-fold-{test_fold}-samples-None', 'best')
            else:
                # This patient region wasn't used for training any models, let's just use the model of the first fold.
                segmenter = (f'segmenter-{region}', 'clinical-fold-0-samples-None', 'best') 

            # Add index row.
            data = {
                'region': region,
                'model': f'clinical-fold-{test_fold}-samples-None'
            }
            index_df = append_row(index_df, data)

            # Save localiser prediction (in normal location) and segmenter prediction (for easy transfer to PMCC).
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', dataset, pat_id, f'{region}.npz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            create_patient_localiser_prediction(dataset, pat_id, localiser, device=device, logging=False)
            create_patient_segmenter_prediction(dataset, pat_id, region, localiser, segmenter, device=device, logging=False, savepath=filepath)

        # Save patient index.
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', dataset, pat_id, 'index.csv')
        index_df.to_csv(filepath, index=False)

def create_two_stage_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    loc_size: types.ImageSize3D = (128, 128, 150),
    n_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)
    log.info(f"Making two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        log.info('Predicting on CPU...')

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(n_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

        # Make predictions.
        for dataset_b, pat_id_b in tqdm(iter(test_loader)):
            if type(pat_id_b) == torch.Tensor:
                pat_id_b = pat_id_b.tolist()
            for dataset, pat_id in zip(dataset_b, pat_id_b):
                create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, device=device, logging=False)
                create_patient_segmenter_prediction(dataset, pat_id, region, localiser.name, segmenter, device=device)
