# Transfer Learning

## Installation

1. Install python (v3.8.2) using [virtualenv](https://virtualenv.pypa.io/en/latest/) or other tool.

2. Install python packages.
```
pip install --requirements requirements.txt
```

3. Set data folder. All training data, models, etc. will live here. Run this command in your bash terminal or add to your profile for something more permanent.
```       
export TL_DATA=<data-dir>
```

## Experiment

1. Download public datasets from TCIA. If using pre-trained models skip to step ... below.
2. Add public datasets to `<data-dir>/datasets/dicom/` as outlined in ...
3. Symlink region maps (see section).
4. Process public DICOM datasets using:

```
from mymi.dataset.dicom import convert_to_nifti

public_datasets = ['HN1', 'HNPCT', 'HNSCC', 'OPC']

# Process datasets - do this in parallel in reality.
for dataset in public_datasets:
    convert_to_nifti(dataset, regions='all', anonymise=False)
```

5. Create public training data for localiser/segmenter networks:

```
from mymi.dataset.nifti import convert_to_training

public_datasets = ['HN1', 'HNPCT', 'HNSCC', 'OPC']
dilate_regions = ['BrachialPlexus_L', 'BrachialPlexus_R', 'Cochlea_L', 'Cochlea_R', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R']

# Process datasets - do this in parallel in reality.
for dataset in public_datasets:
    # Create data for localiser.
    convert_to_training(dataset, 'all', f'{dataset}-LOC', dilate_regions=dilate_regions, size=(128, 128, 150), spacing=(4, 4, 4))

    # Create data for segmenter.
    convert_to_training(dataset, 'all', f'{dataset}-SEG', size=None, spacing=(1, 1, 2))
```

6. Train separate public localiser/segmenter networks per region.

```
from mymi.training.localiser import train_localiser
from mymi.regions import RegionNames

loc_datasets = ['HN1-LOC', 'HNPCT-LOC', 'HNSCC-LOC', 'OPC-LOC']
seg_datasets = ['HN1-SEG', 'HNPCT-SEG', 'HNSCC-SEG', 'OPC-SEG']

# Train localiser/segmenter network per region - in reality this would be performed across multiple machines.
for region in RegionNames:
    # Train localiser network.
    train_localiser(loc_datasets, region, f'localiser-{region}', 'public-1gpu-150epochs', n_epochs=150)

    # Train segmenter network.
    train_segmenter(seg_datasets, region, f'segmenter-{region}', 'public-1gpu-150epochs', n_epochs=150)
```

Training can be resumed upon failure using:

```
from mymi.training.localiser import train_localiser
from mymi.training import train

models = ['localiser', 'segmenter']
datasets = {
    'localiser': ['HN1-LOC', 'HNPCT-LOC', 'HNSCC-LOC', 'OPC-LOC']
    'segmenter': ['HN1-SEG', 'HNPCT-SEG', 'HNSCC-SEG', 'OPC-SEG']
}
region = 'Brain'

# For failed localiser.
train(datasets, region, f'localiser-{region}', 'public-1gpu-150epochs', n_epochs=150, resume=True, resume_checkpoint='last')

# For failed segmenter.
train_segmenter(datasets, region, f'segmenter-{region}', 'public-1gpu-150epochs', n_epochs=150, resume=True, resume_checkpoint='last')
```

def train_localiser(
    model_name: str,
    run_name: str,
    datasets: Union[str, List[str]],
    region: str,
    loss: str = 'dice',
    n_epochs: int = 200,
    n_folds: Optional[int] = None,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    pretrained: Optional[Tuple[str, str, str]] = None,
    p_val: float = 0.2,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
def train_segmenter(
    datasets: Union[str, List[str]],
    region: str,
    model: str,
    run: str,
    loss: str = 'dice',
    n_epochs: int = 200,
    n_folds: Optional[int] = 5,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    pretrained_model: Optional[types.ModelName] = None,    
    p_val: float = 0.2,
    resume: bool = False,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
## Datasets

### DICOM Datasets

#### Setup

To add a DICOM dataset, drop all data into the folder `<data-dir>/datasets/dicom/<dataset>/data` where `<dataset>` is the name of your dataset as it will appear in the `Dataset` API.

Note that *no dataset file structure* is enforced as the indexing engine will traverse the folder, locating all DICOM files, and creating an index (at `<dataset>/index.csv`) that will be used by the `Dataset` API to make queries on the dataset.

#### Index

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

##### Index Exclusion Criteria

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

#### Region maps

If your datasets have inconsistent organ-at-risk (region) names, you can apply a region map to map from many values in the dataset to a single internal name. Many examples are shown in `mymi/dataset/dicom/files/region-maps` for the public datasets from this study.

Add the region map at `<dataset>/region-map.csv`. The map must contain a `dataset` column with region names as they appear in the dataset (regexp capable) and an `internal` column with the internal name to map to. An optional column `case-sensitive` specifies whether the `dataset` column is case sensitive (default=False).

#### API

If there are multiple studies/RTSTRUCTs for a patient, the first study and RTSTRUCT are chosen as the default. To set different defaults, pass `study_index` and `rtstruct_index` kwargs upon patient creation (e.g. `set.patient('<patient-id>', study_index=1, rtstruct_index=2))`. The default CT is always the CT series attached to the default RTSTRUCT series.

##### API - Datasets

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

##### API - Patients

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

##### API - Studies
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')

# List studies.
pat.list_studies()

# Load study.
pat.study('<study-id>')
```

##### API - Series
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

###### API - CT Series
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

###### API - RTSTRUCT Series
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

###### API - RTPLAN Series
```
from mymi import dataset as ds

set = ds.get('<dataset>', 'dicom')
pat = set.patient('<patient-id>')
study = pat.study('<study-id'>)
series = study.series('<series-id>')

# Get pydicom RTPLAN 'FileDataset' object.
series.get_rtplan()
```

###### API - RTDOSE Series
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

### NIFTI Datasets

#### Setup

NIFI datasets can be created by processing an existing DICOM dataset or by manually creating a folder of the correct structure.

##### Processing DICOM

We can process the CT/region data from an existing `DicomDataset` into a `NiftiDataset` using the following command. We can specify the subset of regions which we'd like to include in our `NiftiDataset` and also whether to anonymise patient IDs (for transferral to external system for training/evaluation, e.g. high-performance computing cluster).

```
from mymi.processing.dataset.dicom import convert_to_nifti

convert_to_nifti('<dataset>', regions=['Brain, 'BrainStem'], anonymise=False)
```

When anonymising, a map back to the true patient IDs will be saved in `<data-dir>/files/anon-maps/<dataset>.csv`.

##### Manual Creation

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

#### API

##### API - Datasets

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

##### API - Patients

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

