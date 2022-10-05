from typing import List

from mymi import types

from .colours import to_255, RegionColours
from .dose_constraints import get_dose_constraint
from .limits import RegionLimits
from .patch_sizes import get_region_patch_size
from .regions import RegionNames, Regions
from .tolerances import get_region_tolerance, RegionTolerances

def is_region(name: str) -> bool:
    # Get region names.
    names = [r.name for r in Regions]
    return name in names

def to_list(
    regions: types.PatientRegions,
    all_regions: List[str]) -> List[str]:
    if type(regions) == str:
        if regions == 'all':
            return all_regions
        else:
            return [regions]
    else:
        return regions

