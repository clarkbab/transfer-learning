from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.nifti import convert_segmenter_predictions_to_dicom_aggregate

convert_segmenter_predictions_to_dicom_aggregate(
    n_pats=20,
    use_model_manifest=True)
