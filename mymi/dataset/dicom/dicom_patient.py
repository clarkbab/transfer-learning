import os
import pandas as pd
from typing import Any, Callable, List, Optional

from mymi import types
from mymi.utils import append_row

from .dicom_study import DICOMStudy
from .region_map import RegionMap

class DICOMPatient:
    def __init__(
        self,
        dataset: 'DICOMDataset',
        id: types.PatientID,
        ct_from: Optional['DICOMPatient'] = None,
        load_default_rtstruct: bool = True,
        load_default_rtdose: bool = False,
        region_map: Optional[RegionMap] = None,
        rtstruct_index: int = 0,
        study_index: int = 0,
        trimmed: bool = False):
        if trimmed:
            self._global_id = f"{dataset} - {id} (trimmed)"
        else:
            self._global_id = f"{dataset} - {id}"
        self._ct_from = ct_from
        self._dataset = dataset
        self._id = str(id)
        self._trimmed = trimmed
        self._region_map = region_map

        # Get patient index.
        index = self._dataset.index
        index = index[index['patient-id'] == str(id)]
        self._index = index

        # Check that patient ID exists.
        if len(index) == 0:
            raise ValueError(f"Patient '{self}' not found in index for dataset '{dataset}'.")
        
        if load_default_rtstruct:
            self._load_default_rtstruct(study_index, rtstruct_index)

        if load_default_rtdose:
            self._load_default_rtdose()

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def __str__(self) -> str:
        return self._global_id

    @property
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate

    @property
    def ct_from(self) -> str:
        return self._ct_from

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def id(self) -> str:
        return self._id

    @property
    def default_rtstruct(self) -> str:
        return self._default_rtstruct

    @property
    def default_rtdose(self) -> str:
        return self._default_rtdose

    @property
    def name(self) -> str:
        return self.get_cts()[0].PatientName

    @property
    def sex(self) -> str:
        return self.get_cts()[0].PatientSex

    @property
    def size(self) -> str:
        return getattr(self.get_cts()[0], 'PatientSize', '')

    @property
    def weight(self) -> str:
        return getattr(self.get_cts()[0], 'PatientWeight', '')

    def list_studies(self) -> List[str]:
        studies = list(sorted(self._index['study-id'].unique()))
        return studies

    def study(
        self,
        id: str) -> DICOMStudy:
        return DICOMStudy(self, id, region_map=self._region_map)

    def info(self) -> pd.DataFrame:
        # Define dataframe structure.
        cols = {
            'age': str,
            'birth-date': str,
            'name': str,
            'sex': str,
            'size': str,
            'weight': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        for col in cols.keys():
            col_method = col.replace('-', '_')
            data[col] = getattr(self, col_method)

        # Add row.
        df = append_row(df, data)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df
    
    def _load_default_rtstruct(
        self,
        study_index: int = 0,
        rtstruct_index: int = 0) -> None:
        # Preference the first study - all studies without RTSTRUCTs have been trimmed.
        # TODO: Add configuration to determine which (multiple?) RTSTRUCTs to select.
        study = self.study(self.list_studies()[study_index])
        rt_series = study.series(study.list_series('RTSTRUCT')[rtstruct_index], 'RTSTRUCT')
        self._default_rtstruct = rt_series

    def _load_default_rtdose(self) -> None:
        # Get RTPLAN series linked to the default RTSTRUCT by 'SOPInstanceUID'.
        def_study = self._default_rtstruct.study
        def_rt_sop_id = self._default_rtstruct.get_rtstruct().SOPInstanceUID
        rtplan_series_ids = def_study.list_series('RTPLAN')
        linked_rtplan_sop_ids = []
        for rtplan_series_id in rtplan_series_ids:
            rtplan = def_study.series(rtplan_series_id, 'RTPLAN')
            rtplan_ref_rt_sop_id = rtplan.get_rtplan().ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            if rtplan_ref_rt_sop_id == def_rt_sop_id:
                linked_rtplan_sop_ids.append(rtplan.get_rtplan().SOPInstanceUID)
        if len(linked_rtplan_sop_ids) == 0:
            raise ValueError(f"No RTPLAN linked to default RTSTRUCT for patient '{self}'.") 

        # Select the first RTPLAN and get linked RTDOSE series.
        def_rtplan_sop_id = linked_rtplan_sop_ids[0]
        rtdose_series_ids = def_study.list_series('RTDOSE')
        linked_rtdose_series = []
        for rtdose_series_id in rtdose_series_ids:
            rtdose = def_study.series(rtdose_series_id, 'RTDOSE')
            rtdose_ref_rtplan_sop_id = rtdose.get_rtdose().ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
            if rtdose_ref_rtplan_sop_id == def_rtplan_sop_id:
                linked_rtdose_series.append(rtdose)
        if len(linked_rtdose_series) == 0:
            raise ValueError(f"No RTDOSE linked to default RTPLAN for patient '{self}'.") 

        # Select the first RTDOSE as the default.
        self._default_rtdose = linked_rtdose_series[0]

    # Proxy to default RTSTRUCT series.

    @property
    def ct_offset(self):
        return self._default_rtstruct.ref_ct.offset

    @property
    def ct_size(self):
        return self._default_rtstruct.ref_ct.size

    @property
    def ct_spacing(self):
        return self._default_rtstruct.ref_ct.spacing

    def ct_orientation(self, *args, **kwargs):
        return self._default_rtstruct.ref_ct.orientation(*args, **kwargs)

    def ct_slice_summary(self, *args, **kwargs):
        return self._default_rtstruct.ref_ct.slice_summary(*args, **kwargs)

    def ct_summary(self, *args, **kwargs):
        return self._default_rtstruct.ref_ct.summary(*args, **kwargs)

    @property
    def ct_data(self):
        return self._default_rtstruct.ref_ct.data

    @property
    def dose_data(self):
        return self._default_rtdose.data

    @property
    def dose_offset(self):
        return self._default_rtdose.offset

    @property
    def dose_size(self):
        return self._default_rtdose.size

    @property
    def dose_spacing(self):
        return self._default_rtdose.spacing

    def get_rtdose(self, *args, **kwargs):
        return self._default_rtdose_series.get_rtdose(*args, **kwargs)

    def get_cts(self, *args, **kwargs):
        return self._default_rtstruct.ref_ct.get_cts(*args, **kwargs)

    def get_first_ct(self, *args, **kwargs):
        return self._default_rtstruct.ref_ct.get_first_ct(*args, **kwargs)
 
    def get_rtstruct(self, *args, **kwargs):
        return self._default_rtstruct.get_rtstruct(*args, **kwargs)

    def has_region(self, *args, **kwargs):
        return self._default_rtstruct.has_region(*args, **kwargs)

    def list_regions(self, *args, **kwargs):
        return self._default_rtstruct.list_regions(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self._default_rtstruct.region_data(*args, **kwargs)

    def region_summary(self, *args, **kwargs):
        return self._default_rtstruct.region_summary(*args, **kwargs)
