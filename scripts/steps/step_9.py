from mymi.dataset.nifti import convert_to_training

# Create data for segmenter.
convert_to_training('INST', 'INST', regions='all', size=None, spacing=(1, 1, 2))
