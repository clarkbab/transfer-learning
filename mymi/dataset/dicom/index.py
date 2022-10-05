from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom as dcm
import os
from time import time
from tqdm import tqdm
from typing import Dict, List

from mymi import config
from mymi import logging
from mymi.utils import append_row

def build_index(dataset: str) -> None:
    start = time()

    # Load all dicom files.
    dataset_path = os.path.join(config.directories.datasets, 'dicom', dataset) 
    data_path = os.path.join(dataset_path, 'data')
    if not os.path.exists(data_path):
        raise ValueError(f"No 'data' folder found for dataset '{dataset}'.")

    # Create index.
    index_cols = {
        'patient-id': str,
        'study-id': str,
        'modality': str,
        'series-id': str,
        'sop-id': str,
        'filepath': str,
        'mod-spec': object
    }
    index = pd.DataFrame(columns=index_cols.keys())

    # Add all DICOM files.
    logging.info(f"Building index for dataset '{dataset}'...")
    for root, _, files in tqdm(os.walk(data_path)):
        for f in files:
            # Check if DICOM file.
            filepath = os.path.join(root, f)
            try:
                dicom = dcm.read_file(filepath, stop_before_pixels=True)
            except dcm.errors.InvalidDicomError:
                continue

            # Get modality.
            modality = dicom.Modality
            if not modality in ('CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE'):
                continue

            # Get patient ID.
            pat_id = dicom.PatientID

            # Get study UID.
            study_id = dicom.StudyInstanceUID

            # Get series UID.
            series_id = dicom.SeriesInstanceUID

            # Get SOP UID.
            sop_id = dicom.SOPInstanceUID

            # Get modality-specific info.
            if modality == 'CT':
                if not hasattr(dicom, 'ImageOrientationPatient'):
                    logging.error(f"No 'ImageOrientationPatient' found for CT dicom '{filepath}'.")
                    continue

                mod_spec = {
                    'ImageOrientationPatient': dicom.ImageOrientationPatient,
                    'ImagePositionPatient': dicom.ImagePositionPatient,
                    'InstanceNumber': dicom.InstanceNumber,
                    'PixelSpacing': dicom.PixelSpacing
                }
            elif modality == 'RTDOSE':
                mod_spec = {
                    'RefRTPLANSOPInstanceUID': dicom.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                }
            elif modality == 'RTPLAN':
                mod_spec = {
                    'RefRTSTRUCTSOPInstanceUID': dicom.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                }
            elif modality == 'RTSTRUCT':
                mod_spec = {
                    'RefCTSeriesInstanceUID': dicom.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                }
            else:
                mod_spec = {}

            # Add index entry.
            data = {
                'patient-id': pat_id,
                'study-id': study_id,
                'modality': modality,
                'series-id': series_id,
                'sop-id': sop_id,
                'filepath': filepath,
                'mod-spec': mod_spec,
            }
            index = append_row(index, data)

    # Create errors index.
    errors_cols = index_cols.copy()
    errors_cols['error'] = str
    errors = pd.DataFrame(columns=errors_cols.keys())

    # Remove duplicates by 'SOPInstanceUID'.
    dup_rows = index['sop-id'].duplicated()
    dup = index[dup_rows]
    dup['error'] = 'DUPLICATE'
    errors = append_row(errors, dup)
    index = index[~dup_rows]

    # Check CT slices have standard orientation.
    ct = index[index.modality == 'CT']
    def standard_orientation(m: Dict) -> bool:
        orient = m['ImageOrientationPatient']
        return orient == [1, 0, 0, 0, 1, 0]
    stand_orient = ct['mod-spec'].apply(standard_orientation)
    nonstand_idx = stand_orient[~stand_orient].index
    nonstand = index.loc[nonstand_idx]
    nonstand['error'] = 'NON-STANDARD-ORIENTATION'
    errors = append_row(errors, nonstand)
    index = index.drop(nonstand_idx)

    # Check CT slices have consistent x/y position.
    ct = index[index.modality == 'CT']
    def consistent_xy_position(series: pd.Series) -> bool:
        pos = series.apply(lambda m: pd.Series(m['ImagePositionPatient'][:2]))
        pos = pos.drop_duplicates()
        return len(pos) == 1
    cons_xy = ct[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(consistent_xy_position)
    incons_idx = cons_xy[~cons_xy].index
    incons = index.loc[incons_idx]
    incons['error'] = 'INCONSISTENT-POSITION-XY'
    errors = append_row(errors, incons)
    index = index.drop(incons_idx)

    # Check CT slices have consistent x/y spacing.
    ct = index[index.modality == 'CT']
    def consistent_xy_spacing(series: pd.Series) -> bool:
        pos = series.apply(lambda m: pd.Series(m['PixelSpacing']))
        pos = pos.drop_duplicates()
        return len(pos) == 1
    cons_xy = ct[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(consistent_xy_spacing)
    incons_idx = cons_xy[~cons_xy].index
    incons = index.loc[incons_idx]
    incons['error'] = 'INCONSISTENT-SPACING-XY'
    errors = append_row(errors, incons)
    index = index.drop(incons_idx)

    # Check CT slices have consistent z spacing.
    ct = index[index.modality == 'CT']
    def consistent_z_position(series: pd.Series) -> bool:
        pos = series.apply(lambda m: m['ImagePositionPatient'][2]).sort_values()
        pos = pos.diff().dropna().round(3)
        pos = pos.drop_duplicates()
        return len(pos) == 1
    cons_z = ct.groupby('series-id')['mod-spec'].transform(consistent_z_position)
    incons_idx = cons_z[~cons_z].index
    incons = index.loc[incons_idx]
    incons['error'] = 'INCONSISTENT-SPACING-Z'
    errors = append_row(errors, incons)
    index = index.drop(incons_idx)

    # If multiple RT files included for a series, keep most recent.
    modalities = ['RTSTRUCT', 'RTPLAN', 'RTDOSE']
    for modality in modalities:
        rt = index[index.modality == modality].sort_values('sop-id', ascending=False)
        dup_idx = rt[rt['series-id'].duplicated()].index
        dup = index.loc[dup_idx]
        dup['error'] = 'MULTIPLE-FILES'
        errors = append_row(errors, dup)
        index = index.drop(dup_idx)

    # Check that RTSTRUCT references CT series in index.
    ct_series = index[index.modality == 'CT']['series-id'].unique()
    rtstruct = index[index.modality == 'RTSTRUCT']
    ref_ct = rtstruct['mod-spec'].apply(lambda m: m['RefCTSeriesInstanceUID']).isin(ct_series)
    nonref_idx = ref_ct[~ref_ct].index
    nonref = index.loc[nonref_idx]
    nonref['error'] = 'NO-REF-CT'
    errors = append_row(errors, nonref)
    index = index.drop(nonref_idx)

    # Check that RTPLAN references RTSTRUCT SOP instance in index.
    rtstruct_sops = index[index.modality == 'RTSTRUCT']['sop-id'].unique()
    rtplan = index[index.modality == 'RTPLAN']
    ref_rtstruct = rtplan['mod-spec'].apply(lambda m: m['RefRTSTRUCTSOPInstanceUID']).isin(rtstruct_sops)
    nonref_idx = ref_rtstruct[~ref_rtstruct].index
    nonref = index.loc[nonref_idx]
    nonref['error'] = 'NO-REF-RTSTRUCT'
    errors = append_row(errors, nonref)
    index = index.drop(nonref_idx)

    # Check that RTDOSE references RTPLAN SOP instance in index.
    rtplan_sops = index[index.modality == 'RTPLAN']['sop-id'].unique()
    rtdose = index[index.modality == 'RTDOSE']
    ref_rtplan = rtdose['mod-spec'].apply(lambda m: m['RefRTPLANSOPInstanceUID']).isin(rtplan_sops)
    nonref_idx = ref_rtplan[~ref_rtplan].index
    nonref = index.loc[nonref_idx]
    nonref['error'] = 'NO-REF-RTPLAN'
    errors = append_row(errors, nonref)
    index = index.drop(nonref_idx)

    # Check that study has RTSTRUCT series.
    incl_rows = index.groupby('study-id')['modality'].transform(lambda s: 'RTSTRUCT' in s.unique())
    nonincl = index[~incl_rows]
    nonincl['error'] = 'STUDY-NO-RTSTRUCT'
    errors = append_row(errors, nonincl)
    index = index[incl_rows]

    # Save index.
    logging.info(f"Saving index for dataset '{dataset}'...")
    index = index.astype(index_cols)
    filepath = os.path.join(dataset_path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Save errors index.
    logging.info(f"Saving index errors for dataset '{dataset}'...")
    errors = errors.astype(errors_cols)
    filepath = os.path.join(dataset_path, 'index-errors.csv')
    errors.to_csv(filepath, index=False)

    # Save indexing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    filepath = os.path.join(dataset_path, f'__INDEXING_TIME_MINS_{mins}__')
    Path(filepath).touch()
