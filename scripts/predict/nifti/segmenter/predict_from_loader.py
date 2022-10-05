import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.prediction.dataset.nifti import create_segmenter_predictions_from_loader

fire.Fire(create_segmenter_predictions_from_loader)

# Sample args:
# --dataset PMCC-HN-TEST --localiser "(...)" --loc_size "(...)" --loc_spacing "(...)" --region Parotid_L
