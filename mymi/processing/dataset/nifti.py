from re import I
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import List, Optional, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.dicom import DICOMDataset, ROIData, RTSTRUCTConverter
from mymi.dataset.nifti import NIFTIDataset
from mymi.dataset.training import create, exists, get, recreate
from mymi.loaders import Loader
from mymi import logging
from mymi.models import replace_checkpoint_alias
from mymi.prediction.dataset.nifti import load_patient_segmenter_prediction
from mymi.regions import RegionColours, RegionNames, to_255
from mymi.reporting.loaders import load_loader_manifest
from mymi.transforms import resample_3D, top_crop_or_pad_3D
from mymi import types
from mymi.utils import append_row

def convert_to_training(
    dataset: str,
    regions: types.PatientRegions,
    dest_dataset: str,
    create_data: bool = True,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    recreate_dataset: bool = True,
    round_dp: Optional[int] = None,
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None) -> None:
    if type(regions) == str:
        if regions == 'all':
            regions = RegionNames
        else:
            regions = [regions]

    # Create the dataset.
    if exists(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate(dest_dataset)
        else:
            created = False
            set_t = get(dest_dataset)
            _destroy_flag(set_t, '__CONVERT_FROM_NIFTI_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create(dest_dataset)
    _write_flag(set_t, '__CONVERT_FROM_NIFTI_START__')

    # Notify user.
    logging.info(f"Creating dataset '{set_t}' with recreate_dataset={recreate_dataset}, regions={regions}, dilate_regions={dilate_regions}, dilate_iter={dilate_iter}, size={size} and spacing={spacing}.")

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'regions': [str(regions)],
            'size': [str(size)] if size is not None else ['None'],
            'spacing': [str(spacing)] if spacing is not None else ['None'],
        })
        params_df.to_csv(filepath)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'regions': [str(regions)],
                'size': [str(size)] if size is not None else ['None'],
                'spacing': [str(spacing)] if spacing is not None else ['None'],
            })
            params_df.to_csv(filepath)

    # Load patients.
    set = NIFTIDataset(dataset)
    pat_ids = set.list_patients()

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': str,
        'origin-dataset': str,
        'patient-id': str,
        'region': str
    }
    index = pd.DataFrame(columns=cols.keys())
    for i, pat_id in enumerate(pat_ids):
        # Get patient regions.
        pat_regions = set.patient(pat_id).list_regions()
        
        # Add entries.
        for region in pat_regions:
            data = {
                'dataset': set_t.name,
                'sample-id': i,
                'origin-dataset': set.name,
                'patient-id': pat_id,
                'region': region
            }
            index = append_row(index, data)

    # Write index.
    index = index.astype(cols)
    filepath = os.path.join(set_t.path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat_id in enumerate(tqdm(pat_ids)):
            # Load input data.
            patient = set.patient(pat_id)
            old_spacing = patient.ct_spacing
            input = patient.ct_data

            # Resample input.
            if spacing:
                input = resample_3D(input, old_spacing, spacing)

            # Crop/pad.
            if size:
                # Log warning if we're cropping the FOV as we're losing information.
                if log_warnings:
                    if spacing:
                        fov_spacing = spacing
                    else:
                        fov_spacing = old_spacing
                    fov = np.array(input.shape) * fov_spacing
                    new_fov = np.array(size) * fov_spacing
                    for axis in range(len(size)):
                        if fov[axis] > new_fov[axis]:
                            logging.warning(f"Patient '{patient}' had FOV '{fov}', larger than new FOV after crop/pad '{new_fov}' for axis '{axis}'.")

                # Perform crop/pad.
                input = top_crop_or_pad_3D(input, size, fill=input.min())

            # Save input.
            _create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat_id).has_region(region):
                    continue

                # Load label data.
                label = patient.region_data(regions=region)[region]

                # Resample data.
                if spacing:
                    label = resample_3D(label, old_spacing, spacing)

                # Crop/pad.
                if size:
                    label = top_crop_or_pad_3D(label, size)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    _create_training_label(set_t, i, region, label)

    end = time()

    # Indicate success.
    _write_flag(set_t, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    _print_time(set_t, hours)

def _destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def _write_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def _print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_NIFTI_TIME_HOURS_{hours}__')
    Path(path).touch()

def _create_training_input(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray) -> None:
    # Save the input data.
    filepath = os.path.join(dataset.path, 'data', 'inputs', f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def _create_training_label(
    dataset: 'Dataset',
    index: int,
    region: str,
    data: np.ndarray) -> None:
    # Save the label data.
    filepath = os.path.join(dataset.path, 'data', 'labels', region, f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def convert_segmenter_predictions_to_dicom_aggregate(
    n_pats: int = 20,
    use_model_manifest: bool = False) -> None:
    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI',
        'institution-name': 'PMCC-AI'
    }

    # Load patients - from Mandible as (fold=0) is missing. 
    loader_datasets = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
    df = load_loader_manifest(loader_datasets, 'Mandible', apply_typing=False, test_fold=0)
    df = df.head(n_pats)
    df = df.astype({ 'origin-patient-id': str })
    datasets = df['origin-dataset']
    pat_ids = df['origin-patient-id']

    # Save missing patients.
    cols = {
        'region': str,
        'dataset': str,
        'patient-id': str
    }
    miss_df = pd.DataFrame(columns=cols.keys())

    for dataset, pat_id in tqdm(list(zip(datasets, pat_ids))):
        # Get ROI ID from DICOM dataset.
        nifti_set = NIFTIDataset(dataset)
        pat_id_dicom = nifti_set.patient(pat_id).patient_id
        set_dicom = DICOMDataset(dataset)
        patient_dicom = set_dicom.patient(pat_id_dicom)
        rtstruct_gt = patient_dicom.default_rtstruct.get_rtstruct()
        info_gt = RTSTRUCTConverter.get_roi_info(rtstruct_gt)
        region_map_gt = dict((set_dicom.to_internal(data['name']), id) for id, data in info_gt.items())

        # Create RTSTRUCT.
        cts = patient_dicom.get_cts()
        rtstruct_pred = RTSTRUCTConverter.create_rtstruct(cts, default_rt_info)
        frame_of_reference_uid = rtstruct_gt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

        # Create index to track which model was used to predict each region.
        cols = {
            'region': str,
            'model': str
        }
        index_df = pd.DataFrame(columns=cols.keys())

        for region in RegionNames:
            # Localiser is always 'public' model.
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')

            # Get segmenter that wasn't trained on this patient.
            for test_fold in range(5):
                # This file has a different format - so we can't trust the loader matches the trained models.
                if region == 'Mandible' and test_fold == 0:
                    continue
                df = load_loader_manifest(loader_datasets, region, test_fold=test_fold)
                df = df[(df.dataset == dataset) & (df['patient-id'] == pat_id)]
                if len(df) == 1:
                    break

            # Check if patient wasn't found in loader.
            if len(df) == 0:
                data = {
                    'region': region,
                    'dataset': dataset,
                    'patient-id': pat_id
                }
                miss_df = append_row(miss_df, data)
                # Skip to next region.
                continue

                # raise ValueError(f"NIFTI patient '({dataset}, {pat_id})' not found in loader manifest for region '{region}'.")

            # Use segmenter that wasn't trained on final 'test_fold' value.
            segmenter = (f'segmenter-{region}', f'clinical-fold-{test_fold}-samples-None', 'best')

            # Add index entry.
            data = {
                'region': region,
                'model': segmenter[1]
            }
            index_df = append_row(index_df, data)

            # Load prediction.
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter, use_model_manifest=use_model_manifest)
            
            # Match ROI number to ground truth if available.
            if region in region_map_gt:
                number = region_map_gt[region]
            else:
                for i in range(1, 1000):
                    if i not in region_map_gt[region].values():
                        region_map_gt[region] = i
                        break
                    elif i == 999:
                        raise ValueError(f'Unlikely')
                number = i 

            # Add ROI data.
            roi_data = ROIData(
                colour=list(to_255(getattr(RegionColours, region))),
                data=pred,
                frame_of_reference_uid=frame_of_reference_uid,
                name=region,
                number=number
            )
            RTSTRUCTConverter.add_roi(rtstruct_pred, roi_data, cts)

        # Save prediction.
        # Hack - clean up when/if path limits are removed.
        if os.environ['PETER_MAC_HACK'] == 'True':
            if dataset == 'PMCC-HN-TEST':
                pred_path = 'S:\\ImageStore\\AtlasSegmentation\\BC_HN\\dicom\\test'
            elif dataset == 'PMCC-HN-TRAIN':
                pred_path = 'S:\\ImageStore\\AtlasSegmentation\\BC_HN\\dicom\\train'
        else:
            pred_path = os.path.join(nifti_set.path, 'predictions', 'segmenter')
        filepath = os.path.join(pred_path, 'aggregate', pat_id_dicom, 'pred.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct_pred.save_as(filepath)

        # Copy CTs.
        for i, path in enumerate(patient_dicom.default_rtstruct.ref_ct.paths):
            filepath = os.path.join(pred_path, 'aggregate', pat_id_dicom, f'ct-{i}.dcm')
            shutil.copyfile(path, filepath)

        # Copy ground truth RTSTRUCT.
        filepath = os.path.join(pred_path, 'aggregate', pat_id_dicom, 'rtstruct.dcm')
        shutil.copyfile(patient_dicom.default_rtstruct.path, filepath)

        # Save index.
        filepath = os.path.join(pred_path, 'aggregate', pat_id_dicom, 'index.csv')
        index_df.to_csv(filepath, index=False)

        # Save missing patient regions.
        filepath = os.path.join(pred_path, 'aggregate', pat_id_dicom, 'missing.csv')
        miss_df.to_csv(filepath, index=False)

def convert_segmenter_predictions_to_dicom_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_manifest: bool = False,
    use_model_manifest: bool = False) -> None:
    # Get unique name.
    localiser = replace_checkpoint_alias(*localiser, use_manifest=use_model_manifest)
    segmenter = replace_checkpoint_alias(*segmenter, use_manifest=use_model_manifest)
    logging.info(f"Converting segmenter predictions to DICOM for '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Build test loader.
    if use_loader_manifest:
        man_df = load_loader_manifest(datasets, region, n_folds=n_folds, test_fold=test_fold)
        samples = man_df[['dataset', 'patient-id']].to_numpy()
    else:
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
        test_dataset = test_loader.dataset
        samples = [test_dataset.__get_item(i) for i in range(len(test_dataset))]

    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI',
        'institution-name': 'PMCC-AI'
    }

    # Create prediction RTSTRUCTs.
    for dataset, pat_id_nifti in tqdm(samples):
        # Get ROI ID from DICOM dataset.
        nifti_set = NIFTIDataset(dataset)
        pat_id_dicom = nifti_set.patient(pat_id_nifti).patient_id
        set_dicom = DICOMDataset(dataset)
        patient_dicom = set_dicom.patient(pat_id_dicom)
        rtstruct_gt = patient_dicom.default_rtstruct.get_rtstruct()
        info_gt = RTSTRUCTConverter.get_roi_info(rtstruct_gt)
        region_map_gt = dict((set_dicom.to_internal(data['name']), id) for id, data in info_gt.items())

        # Create RTSTRUCT.
        cts = patient_dicom.get_cts()
        rtstruct_pred = RTSTRUCTConverter.create_rtstruct(cts, default_rt_info)

        # Load prediction.
        pred = load_patient_segmenter_prediction(dataset, pat_id_nifti, localiser, segmenter)
        
        # Add ROI.
        roi_data = ROIData(
            colour=list(to_255(getattr(RegionColours, region))),
            data=pred,
            frame_of_reference_uid=rtstruct_gt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
            name=region,
            number=region_map_gt[region]        # Patient should always have region (right?) - we created the loaders based on patient regions.
        )
        RTSTRUCTConverter.add_roi(rtstruct_pred, roi_data, cts)

        # Save prediction.
        # Get localiser checkpoint and raise error if multiple.
        # Hack - clean up when/if path limits are removed.
        if config.environ('PETER_MAC_HACK') == 'True':
            base_path = 'S:\\ImageStore\\HN_AI_Contourer\\short\\dicom'
            if dataset == 'PMCC-HN-TEST':
                pred_path = os.path.join(base_path, 'test')
            elif dataset == 'PMCC-HN-TRAIN':
                pred_path = os.path.join(base_path, 'train')
        else:
            pred_path = os.path.join(nifti_set.path, 'predictions', 'segmenter')
        filepath = os.path.join(pred_path, *localiser, *segmenter, f'{pat_id_dicom}.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct_pred.save_as(filepath)
