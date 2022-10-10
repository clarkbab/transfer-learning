# Transfer Learning

## Installation

1. Install python (v3.8.2) using [virtualenv](https://virtualenv.pypa.io/en/latest/) or other tool.
2. Install python packages.
```
pip install -r requirements.txt
```

3. Set data folder. All training data, models, etc. will live here. Run this command in your bash terminal or add to your profile for something more permanent.
```       
export TL_DATA=<data-dir>
```

## Transfer Learning Experiment

The preferred method for running this experiment is with access to a Slurm-managed high-performance computing cluster. Scripts are provided in [scripts/slurm](scripts/slurm) to create jobs with minimal editing of scripts required - please see instructions in files. Otherwise, python scripts can be found in [scripts/steps](scripts/steps) and should be modified to run on your preferred training platform.

Note that all scripts should be run from the root project folder (e.g. `python scripts/slurm/step_4/create_jobs.py`).

### Transfer Learning Experiment - Steps

1. Download TCIA datasets. If using pre-trained models skip to step ... below.
- [Head-Neck-Radiomics-HN1](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-Radiomics-HN1)
- [Head-Neck-PET-CT](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT)
- [HNSCC](https://wiki.cancerimagingarchive.net/display/Public/HNSCC)
- [OPC-Radiomics](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=33948764)
2. Create DICOM datasets from public datasets as outlined in [DICOM Dataset - Setup](#dicom-dataset-setup). These should be named HN1, HNPCT, HNSCC and OPC.
3. Symlink region maps as outlined in [DICOM Dataset - Region Maps](#dicom-dataset-region-maps).
4. Process public DICOM datasets using the [slurm script](scripts/slurm/step_4/create_jobs.py) (creates 4 jobs) or by modifying the [python code](scripts/steps/step_4.py).
5. Create public training data for localiser/segmenter networks using the [slurm script](scripts/slurm/step_5/create_jobs.py) (creates 4 jobs) or by modifying the [python code](scripts/steps/step_5.py).
6. Train separate public localiser/segmenter networks per region using the [slurm script](scripts/slurm/step_6/create_jobs.py) (creates 34 jobs; 2 array jobs of length 17) or by modifying the [python code](scripts/steps/step_6.py).
- Training can be resumed upon failure using the [slurm script](scripts/slurm/step_6/create_resume_job.py) (creates 1 job) or by modifying the [python code](scripts/steps/step_6_resume.py).
7. Create the institutional ([NIFTI](#nifti-dataset-setup) or [DICOM](#dicom-dataset-setup)) dataset with name 'INST' from your institutional dataset or another TCIA dataset.
8. Process DICOM dataset to NIFTI if required using the [slurm script](scripts/slurm/step_8/create_job.py) (creates 1 job) or by modifying the [python code](scripts/steps/step_8.py).
9. Create institutional training data for segmenter networks using the [slurm script](scripts/slurm/step_9/create_job.py) (creates 1 job) or by modifyting the [python code](scripts/steps/step_9.py).
10. Train separate institutional segmenter networks per region for increasing sample sizes (e.g. n=5,10,20,50,...,'max') using the [slurm script](scripts/slurm/step_10/create_jobs.py) (creates x jobs) or by modifying the [python code](scripts/steps/step_10.py).
11. This step requires the completion of public segmenter training (step 6). Train separate transfer segmenter networks per region for increasing sample sizes (e.g. n=5,10,20,50,...,'max') using the [slurm script](scripts/slurm/step_11/create_jobs.py) (creates x jobs) or by modifying the [python code](scripts/steps/step_11.py).

## Datasets

### DICOM Dataset

#### DICOM Dataset - Setup

To add a DICOM dataset, drop all data into the folder `<data-dir>/datasets/dicom/<dataset>/data` where `<dataset>` is the name of your dataset as it will appear in the `Dataset` API.

Note that *no dataset file structure* is enforced as the indexing engine will traverse the folder, locating all DICOM files, and creating an index (at `<dataset>/index.csv`) that will be used by the `Dataset` API to make queries on the dataset.

#### DICOM Dataset - Index

The index is built when a dataset is first used via the `Dataset` API. Indexing can also be triggered via the command:

```
from mymi.dataset.dicom import build_index
build_index('<dataset>')
```

The index contains a hierarchy of objects that can be queried using the `Dataset` API. During building of the index, some objects may be excluded if they don't meet the inclusion criteria, e.g. a patient will be excluded from the index if they don't have valid CT/RTSTRUCT series. All excluded objects are stored in `<dataset>/index-errors.csv`.

The index object hierarchy is:

```
- <dataset> 
    - <patient 1>
        - <study 1>
            - <series 1> (e.g. CT)
            - <series 2> (e.g. RTSTRUCT)
        - <study 2>
            ...
    - <patient 2>
        ...
```

##### DICOM Dataset - Index - Exclusion Criteria

The following rules are applied *in order* to exclude objects from the index. All excluded objects are saved in `index-errors.csv` with the applicable error code.

Rules can be switched off with a `<dataset>/index-rules.csv` file. The file must contain columns `code` with rule codes (e.g. `DUPLICATE`) and `apply` with boolean values (e.g. `false`).

Order | Code | Description
--- | --- | ---
1 | DUPLICATE | Duplicate DICOM files are removed.<br/>Duplicates are determined by DICOM field 'SOPInstanceUID'
2 | NON-STANDARD-ORIENTATION | CT DICOM series with non-standard orientation are removed.<br/>DICOM field 'ImageOrientationPatient' is something other than `[1, 0, 0, 0, 1, 0]`
3 | INCONSISTENT-POSITION-XY | CT DICOM series with inconsistent x/y position across slices are removed.<br/>DICOM field 'ImagePositionPatient' x/y elements should be consistent.
4 | INCONSISTENT-SPACING-XY | CT DICOM series with inconsistent x/y spacing across slices are removed.<br/>DICOM field 'PixelSpacing' should be consistent.
5 | INCONSISTENT-SPACING-Z | CT DICOM series with inconsistent z spacing are removed.<br/>Difference between DICOM field 'ImagePositionPatient' z position for slices (sorted by z position) should be consistent.
6 | MULTIPLE-FILES | Duplicate RTSTRUCT/RTPLAN/RTDOSE files for a series are removed.<br/>First RTSTRUCT/RTPLAN/RTDOSE are retrained for a series (ordered by DICOM field 'SOPInstanceUID')
7 | NO-REF-CT | RTSTRUCT DICOM series without a referenced CT series are removed.<br/>CT series referenced by RTSTRUCT DICOM field `ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID` should be present in the index.
8 | NO-REF-RTSTRUCT | RTPLAN DICOM series without a referenced RTSTRUCT series are removed.<br/>RTSTRUCT series referenced by RTPLAN DICOM field `ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID` should be present in the index.
9 | NO-REF-RTPLAN | RTDOSE DICOM series without a referenced RTPLAN series are removed.<br/>RTPLAN series referenced by RTDOSE DICOM field `ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID` should be present in the index.
10 | NO-RTSTRUCT | Studies without RTSTRUCT series in index are removed.

Feel free to add/remove/update these criteria by modifying the file at `mymi/dataset/dicom/index.py`.

#### DICOM Dataset - Region Maps

If your datasets have inconsistent organ-at-risk (region) names, you can apply a region map to map from many values in the dataset to a single internal name. Many examples are shown in `mymi/dataset/dicom/files/region-maps` for the public datasets from this study.

Add the region map at `<dataset>/region-map.csv`. The map must contain a `dataset` column with region names as they appear in the dataset (regexp capable) and an `internal` column with the internal name to map to. An optional column `case-sensitive` specifies whether the `dataset` column is case sensitive (default=False).

#### DICOM Dataset - API

If there are multiple studies/RTSTRUCTs for a patient, the first study and RTSTRUCT are chosen as the default. To set different defaults, pass `study_index` and `rtstruct_index` kwargs upon patient creation (e.g. `set.patient('<patient-id>', study_index=1, rtstruct_index=2))`. The default CT is always the CT series attached to the default RTSTRUCT series.

##### DICOM Dataset - API - Datasets

```
from mymi import dataset as ds
from mymi.dataset.dicom import DicomDataset

# List datasets.
ds.list_datasets()

# Load dataset.
set = ds.get('<dataset>', 'dicom')
set = ds.get('<dataset>')           # Will raise an error if there are multiple datasets with this name.
set = DicomDataset('<dataset>')     # Using constructor directly.
```

##### DICOM Dataset - API - Patients

```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')

# List patients.
set.list_patients()
set.list_patients(regions=['Brain', 'BrainStem'])       # List patients who have certain regions. Slow query as it must read RTSTRUCT file associated with all patients.

# Check for patient.
set.has_patient('<patient-id>')

# Load patient.
pat = set.patient('<patient-id>')

# List patient regions (using default CT/RTSTRUCT series).
pat.list_regions()

# Get CT/region data (using default CT/RTSTRUCT series).
pat.ct_data
pat.region_data(regions=['Brain', 'BrainStem'])
```

##### DICOM Dataset - API - Studies
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')

# List studies.
pat.list_studies()

# Load study.
pat.study('<study-id>')
```

##### DICOM Dataset - API - Series
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)

# List series of modality 'ct', 'rtstruct', 'rtplan' or 'rtdose'.
study.list_series('ct')
study.list_series('rtstruct')

# Load series.
study.series('<series-id>')
```

###### DICOM Dataset - API - CT Series
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Load CT data.
series.data

# Load CT geometry.
series.size
series.spacing
series.orientation

# Get pydicom CT 'FileDataset' objects.
series.get_cts()
series.get_first_ct()       # If reading duplicated fields (e.g. PatientName) just load the first one.
```

###### DICOM Dataset - API - RTSTRUCT Series
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# List regions.
series.list_regions()

# Check for region.
series.has_region('Brain')

# Load RTSTRUCT region data.
series.region_data(regions=['Brain', 'BrainStem'])

# Get pydicom RTSTRUCT 'FileDataset' object.
series.get_rtstruct()
```

###### DICOM Dataset - API - RTPLAN Series
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Get pydicom RTPLAN 'FileDataset' object.
series.get_rtplan()
```

###### DICOM Dataset - API - RTDOSE Series
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Load RTDOSE data.
series.data

# Load RTDOSE geometry.
series.size
series.spacing
series.orientation

# Get pydicom RTDOSE 'FileDataset' object.
series.get_rtdose()
```

### NIFTI Dataset

#### NIFTI Dataset - Setup

NIFI datasets can be created by processing an existing DICOM dataset or by manually creating a folder of the correct structure.

##### NIFTI Dataset - Setup - Processing DICOM

We can process the CT/region data from an existing `DicomDataset` into a `NiftiDataset` using the following command. We can specify the subset of regions which we'd like to include in our `NiftiDataset` and also whether to anonymise patient IDs (for transferral to external system for training/evaluation, e.g. high-performance computing cluster).

```
from mymi.processing.dataset.dicom import convert_to_nifti

convert_to_nifti('<dataset>', regions=['Brain, 'BrainStem'], anonymise=False)
```

When anonymising, a map back to the true patient IDs will be saved in `<data-dir>/files/anon-maps/<dataset>.csv`.

##### NIFTI Dataset - Setup - Manual Creation

NIFTI datasets can be created by adding CT and region NIFTI files to a folder `<data-dir>/datasets/dicom/<dataset>/data` with the following structure:

```
<dataset> (e.g. 'MyDataset')
    data
        ct
            - <patient 1>.nii (e.g. 0.nii)
            - <patient 2>.nii (e.g. 1.nii)
            - ...
        regions
            <region 1> (e.g. BrachialPlexus_L)
                - <patient 1>.nii
                - <patient 2>.nii
                - ...
            <region 2> (e.g. BrachialPlexus_R)
                - ...
```

#### NIFTI Dataset - API

##### NIFTI Dataset - API - Datasets

```
from mymi import dataset as ds
from mymi.dataset.nifti import NiftiDataset

# List datasets.
ds.list_datasets()

# Load dataset.
set = ds.get('<dataset>', 'nifti')
set = ds.get('<dataset>')           # Will raise an error if there are multiple datasets with this name.
set = NiftiDataset('<dataset>')     # Using constructor directly.
```

##### NIFTI Dataset - API - Patients

```
from mymi import dataset as ds
set = ds.get('<dataset>', 'nifti')

# List patients.
set.list_patients()
set.list_patients(regions=['Brain', 'BrainStem'])       # List patients who have certain regions. Fast query as it's just reading filenames.

# Check for patient.
set.has_patient('<patient-id>')

# Load patient.
pat = set.patient('<patient-id>')

# List patient regions.
pat.list_regions()

# Load CT/region data.
pat.ct_data
pat.region_data(regions=['Brain', 'BrainStem'])

# Load CT geometry.
pat.size
pat.spacing
pat.offset

# Get de-anonymised patient ID. Anon map must be present.
pat.origin
```

### TRAINING datasets

#### Setup

A `TrainingDataset` must be created by running a processing script on an existing `NiftiDataset`. For example:

```
from mymi.processing.dataset.nifti import convert_to_training

convert_to_training(
    '<nifti-dataset>',              # Source dataset name.
    ['Brain', 'BrainStem'],         # Regions to process.
    '<training-dataset>',           # Target dataset name.
    dilate_iter=3,                  # Number of rounds of dilation to perform to 'dilate_regions'.
    dilate_regions=['BrainStem'],   # Regions to dilate (e.g. for localiser training)
    size=(100, 100, 100),           # Crop processed images/labels to this size.
    spacing=(4, 4, 4)               # Resample images/labels to the spacing.
)
```

### Dataset API

The `Dataset` API allows for basic queries to be performed on the installed datasets.

#### DICOM Dataset


#### NIFTI Dataset

#### TRAINING Dataset

## Visualisation




## Pretrained public models

These models were trained on the following Cancer Imaging Archive (TCIA) datasets:
- HN1
- HNPCT
- HNSCC
- OPC

You can download the pretrained public localiser

## Training public model

