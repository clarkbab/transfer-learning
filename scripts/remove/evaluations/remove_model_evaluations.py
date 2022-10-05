import os
from os.path import dirname as up
import pandas as pd
import pathlib
from tqdm import tqdm
import shutil
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config
from mymi.loaders import Loader
from mymi.models.systems import Localiser, Segmenter
from mymi.regions import RegionNames

for_real = True
datasets = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
regions = RegionNames
models = ['clinical', 'public', 'transfer']
n_folds = 5
n_trains = [5, 10, 20, 50, 100, 200, None]
test_folds = list(range(5))

# Load localiser checkpoints - localiser is shared by segmenter stage models.
loc_ckpts = {}
for region in regions:
    ckpt = Localiser.replace_checkpoint_aliases(f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')[2]
    loc_ckpts[region] = ckpt

for region in regions:
    for test_fold in test_folds:
        # Get test cases per dataset.
        tl, vl, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
        n_train_max = len(tl) + len(vl)

        # Remove public model run evaluations.
        if 'public' in models:
            run = 'public-1gpu-150epochs'
            print(f'region:{region}, model:{run}')

            # Delete evaluation.
            eval_path = os.path.join(config.directories.evaluations, 'segmenter', f'localiser-{region}', 'public-1gpu-150epochs', loc_ckpts[region], f'segmenter-{region}', run)
            if os.path.exists(eval_path):
                if for_real:
                    print(f'\t{eval_path}')
                    shutil.rmtree(eval_path)
                else:
                    print(f'\t{eval_path}')
        
        # Remove evaluations for other models.
        for model in (m for m in models if m != 'public'):
            for n_train in n_trains:
                # Check that number of training cases are available.
                if n_train is not None and n_train > n_train_max:
                    continue

                # Get model run.
                run = f'{model}-fold-{test_fold}-samples-{n_train}'
                print(f'region:{region}, model:{run}')

                # Delete evaluation.
                eval_path = os.path.join(config.directories.evaluations, 'segmenter', f'localiser-{region}', 'public-1gpu-150epochs', loc_ckpts[region], f'segmenter-{region}', run)
                if os.path.exists(eval_path):
                    if for_real:
                        print(f'\t{eval_path}')
                        shutil.rmtree(eval_path)
                    else:
                        print(f'\t{eval_path}')
