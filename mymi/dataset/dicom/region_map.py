import numpy as np
import os
import pandas as pd
import re

from mymi import config
from mymi import regions

class RegionMap:
    def __init__(
        self,
        data: pd.DataFrame):
        """
        args:
            data: the mapping data.
        """
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def to_internal(
        self,
        region: str) -> str:
        """
        returns: the internal region name if appropriate mapping was supplied. If no mapping
            was supplied for the region then it remains unchanged.
        args:
            region: the region name to map.
        """
        # Iterrate over map rows.
        for _, row in self._data.iterrows():
            # Create pattern match args.
            args = [row.dataset, region]
            
            # Check case.
            case_sensitive = row['case-sensitive'] if 'case-sensitive' in row else False
            if not np.isnan(case_sensitive) and not case_sensitive:
                args += [re.IGNORECASE]
                
            # Perform match.
            if re.match(*args):
                return row.internal
        
        return region
