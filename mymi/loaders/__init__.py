from typing import List, Optional, Union
from .loader import Loader
from .patch_loader import PatchLoader

def get_loader_n_train(
    datasets: Union[str, List[str]],
    region: str,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> int:
    tl, vl, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
    n_train = len(tl) + len(vl)
    return n_train
