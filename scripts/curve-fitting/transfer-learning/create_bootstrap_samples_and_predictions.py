import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.curve_fitting.transfer_learning import create_bootstrap_samples_and_predictions

fire.Fire(create_bootstrap_samples_and_predictions)

# Sample args:
# --dataset PMCC-HN-TEST --region Parotid_L --segmenter "(...)"
