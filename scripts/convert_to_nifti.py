import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from mymi.processing.dataset.dicom import convert_to_nifti

if __name__ == '__main__':
    fire.Fire(convert_to_nifti)
