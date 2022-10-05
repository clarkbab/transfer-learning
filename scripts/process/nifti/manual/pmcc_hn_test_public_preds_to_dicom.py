from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.nifti import convert_segmenter_predictions_to_dicom

dataset = 'PMCC-HN-TEST'
regions = ('BrachialPlexus_L','BrachialPlexus_R','Brain','BrainStem','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Mandible','OpticNerve_L','OpticNerve_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R')
loc_runs = 'public-1gpu-150epochs'
seg_runs = 'public-1gpu-150epochs'
model = 'public'

convert_segmenter_predictions_to_dicom(dataset, regions, loc_runs, seg_runs, model)
