import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.dicom import convert_to_nifti

fire.Fire(convert_to_nifti)

# Sample args:
# --dataset HEAD-NECK-RADIOMICS-HN1 --regions all --anonymise True
