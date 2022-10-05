import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.evaluation.dataset.nifti import create_segmenter_evaluation_from_loader

fire.Fire(create_segmenter_evaluation_from_loader)

# Sample args:
# --dataset PMCC-HN-TEST --region Parotid_L --segmenter "(...)"
