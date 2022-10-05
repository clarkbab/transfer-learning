import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

# Commented due to circular import.
# from mymi.loaders import Loader
from mymi import config
from mymi import logging

def append_row(df: pd.DataFrame, data: Dict[str, Union[str, int, float]]) -> pd.DataFrame:
    return pd.concat((df, pd.DataFrame(data, index=[0])), axis=0)

def encode(o: Any) -> str:
    return hashlib.sha1(json.dumps(o).encode('utf-8')).hexdigest()

# Commented due to circular import.
# def get_manifest():
#     datasets = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
#     region = 'BrainStem'
#     n_folds = 5
#     n_train = 5
#     test_fold = 0
#     _, _, test_loader = Loader.build_loaders(datasets, region, load_test_origin=False, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
#     samples = []
#     for ds_b, samp_b in iter(test_loader):
#         samples.append((ds_b[0], samp_b[0].item()))
#     return samples

def get_batch_centroids(label_batch, plane):
    """
    returns: the centroid location of the label along the plane axis, for each
        image in the batch.
    args:
        label_batch: the batch of labels.
        plane: the plane along which to find the centroid.
    """
    assert plane in ('axial', 'coronal', 'sagittal')

    # Move data to CPU.
    label_batch = label_batch.cpu()

    # Determine axes to sum over.
    if plane == 'axial':
        axes = (0, 1)
    elif plane == 'coronal':
        axes = (0, 2)
    elif plane == 'sagittal':
        axes = (1, 2)

    centroids = np.array([], dtype=np.int)

    # Loop through batch and get centroid for each label.
    for label_i in label_batch:
        # Get weighting along 'plane' axis.
        weights = label_i.sum(axes)

        # Get average weighted sum.
        indices = np.arange(len(weights))
        avg_weighted_sum = (weights * indices).sum() /  weights.sum()

        # Get centroid index.
        centroid = np.round(avg_weighted_sum).long()
        centroids = np.append(centroids, centroid)

    return centroids

def fplot(
    f_str: str, 
    figsize: Tuple[float, float] = (8, 6),
    x: Optional[List[float]] = None,
    y: Optional[List[float]] = None, 
    xres: float = 1e-1,
    xlim: Tuple[float, float] = (-10, 10),
    **kwargs) -> None:
    # Rename x so it can be used in 'eval'.
    x_data, y_data = x, y
    
    # Replace params in 'f'.
    f = f_str
    params = dict(((k, v) for k, v in kwargs.items() if len(k) == 1 and k not in ('x', 'y')))
    for k, v in params.items():
        f = f.replace(k, str(v))

    # Plot function.
    x = np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) / xres))
    y = eval(f)
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    
    # Plot points.
    if x_data is not None or y_data is not None:
        assert x_data is not None and y_data is not None
        assert len(x_data) == len(y_data)
        plt.scatter(x_data, y_data, marker='x')
        
    param_str = ','.join((f'{k}={v:.3f}' for k, v in params.items()))
    plt.title(f"{f_str}, {param_str}")

    plt.show()

def save_csv(
    data: pd.DataFrame,
    *path: List[str],
    index: bool = False,
    overwrite: bool = False) -> None:
    filepath = os.path.join(config.directories.files, *path)
    dirpath = os.path.dirname(filepath)
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(dirpath, exist_ok=True)
            data.to_csv(filepath, index=index)
        else:
            logging.error(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(dirpath, exist_ok=True)
        data.to_csv(filepath, index=index)

def load_csv(
    *path: List[str],
    raise_error: bool = True,
    **kwargs: Dict[str, str]) -> Optional[pd.DataFrame]:
    filepath = os.path.join(config.directories.files, *path)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, **kwargs)
    elif raise_error:
        raise ValueError(f"CSV at path '{path}' not found.")
    else:
        return None
