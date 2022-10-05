import collections
import numpy as np
import pandas as pd
import pydicom as dcm
from scipy.ndimage import center_of_mass
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

from mymi import types

from .ct_series import CTSeries
from .dicom_series import DICOMModality, DICOMSeries
from .region_map import RegionMap
from .rtstruct_converter import RTSTRUCTConverter

class RTSTRUCTSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str,
        load_ref_ct: bool = True,
        region_map: Optional[RegionMap] = None):
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id
        self.__region_map = region_map

        # Get index.
        index = self._study.index
        index = index[(index.modality == 'RTSTRUCT') & (index['series-id'] == id)]
        self._index = index
        self._path = index.iloc[0]['filepath']      # Select first RTSTRUCT from series as default.

        # Check that series exists.
        if len(index) == 0:
            raise ValueError(f"RTSTRUCT series '{self}' not found in index for study '{study}'.")

        # Load reference CT series.
        if load_ref_ct:
            rtstruct = self.get_rtstruct()
            ct_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
            self._ref_ct = CTSeries(study, ct_id)

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTSTRUCT

    @property
    def path(self) -> str:
        return self._path

    @property
    def ref_ct(self) -> str:
        return self._ref_ct

    @property
    def study(self) -> str:
        return self._study

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    def __str__(self) -> str:
        return self._global_id

    def list_regions(
        self,
        use_mapping: bool = True) -> List[str]:
        rtstruct = self.get_rtstruct()
        names = list(sorted(RTSTRUCTConverter.get_roi_names(rtstruct)))
        names = list(filter(lambda n: RTSTRUCTConverter.has_roi_data(rtstruct, n), names))
        if use_mapping and self.__region_map:
            names = [self.__region_map.to_internal(n) for n in names]
        return names

    def get_rtstruct(self) -> dcm.dataset.FileDataset:
        filepath = self._index.iloc[0].filepath
        rtstruct = dcm.read_file(filepath)
        return rtstruct

    def get_region_info(
        self,
        use_mapping: bool = True) -> Dict[int, Dict[str, str]]:
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region IDs.
        roi_info = RTSTRUCTConverter.get_roi_info(rtstruct)

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        roi_info = dict(filter(lambda i: RTSTRUCTConverter.has_roi_data(rtstruct, i[1]['name']), roi_info.items()))

        # Map to internal names.
        if use_mapping and self.__region_map:
            def map_name(info):
                info['name'] = self.__region_map.to_internal(info['name'])
                return info
            roi_info = dict((id, map_name(info)) for id, info in roi_info.items())

        return roi_info

    def list_regions(
        self,
        use_mapping: bool = True,
        whitelist: types.PatientRegions = 'all') -> List[str]:
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region names.
        names = list(sorted(RTSTRUCTConverter.get_roi_names(rtstruct)))

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        names = list(filter(lambda n: RTSTRUCTConverter.has_roi_data(rtstruct, n), names))

        # Map to internal names.
        if use_mapping and self.__region_map:
            names = [self.__region_map.to_internal(n) for n in names]

        # Filter on whitelist.
        def filter_fn(region):
            if isinstance(whitelist, str):
                if whitelist == 'all':
                    return True
                else:
                    return region == whitelist
            else:
                if region in whitelist:
                    return True
                else:
                    return False
        names = list(filter(filter_fn, names))

        return names

    def has_region(
        self,
        region: str,
        use_mapping: bool = True) -> bool:
        return region in self.list_regions(use_mapping=use_mapping)

    def region_data(
        self,
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> OrderedDict:
        self._assert_requested_regions(regions, use_mapping=use_mapping)

        # Get region names - include unmapped as we need these to load RTSTRUCT regions later.
        unmapped_names = self.list_regions(use_mapping=False)
        names = self.list_regions(use_mapping=use_mapping)
        names = list(zip(names, unmapped_names))

        # Filter on requested regions.
        def fn(pair):
            name, _ = pair
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return name == regions
            else:
                return name in regions
        names = list(filter(fn, names))

        # Get reference CTs.
        cts = self._ref_ct.get_cts()

        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Add ROI data.
        region_dict = {}
        for name, unmapped_name in names:
            data = RTSTRUCTConverter.get_roi_data(rtstruct, unmapped_name, cts)
            region_dict[name] = data

        # Create ordered dict.
        ordered_dict = collections.OrderedDict((n, region_dict[n]) for n in sorted(region_dict.keys())) 

        return ordered_dict

    def _filter_on_dict_keys(
        self,
        keys: Union[str, Sequence[str]] = 'all',
        whitelist: Union[str, List[str]] = None) -> Callable[[Tuple[str, Any]], bool]:
        """
        returns: a function that filters out unneeded keys.
        kwargs:
            keys: description of required keys.
            whitelist: keys that are never filtered.
        """
        def fn(item: Tuple[str, Any]) -> bool:
            key, _ = item
            # Allow based on whitelist.
            if whitelist is not None:
                if type(whitelist) == str:
                    if key == whitelist:
                        return True
                elif key in whitelist:
                    return True
            
            # Filter based on allowed keys.
            if ((isinstance(keys, str) and (keys == 'all' or key == keys)) or
                ((isinstance(keys, list) or isinstance(keys, np.ndarray) or isinstance(keys, tuple)) and key in keys)):
                return True
            else:
                return False
        return fn

    def _assert_requested_regions(
        self,
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> None:
        if type(regions) == str:
            if regions != 'all' and not self.has_region(regions, use_mapping=use_mapping):
                raise ValueError(f"Requested region '{regions}' not present for RTSTRUCT series '{self}'.")
        elif hasattr(regions, '__iter__'):
            for region in regions:
                if not self.has_region(region, use_mapping=use_mapping):
                    raise ValueError(f"Requested region '{region}' not present for RTSTRUCT series '{self}'.")
        else:
            raise ValueError(f"Requested regions '{regions}' isn't 'str' or 'iterable'.")
