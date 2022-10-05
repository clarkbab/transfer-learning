import os
from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(filepath)))
sys.path.append(mymi_dir)
from mymi import config
from mymi.regions import RegionNames

dry_run = True
keep = 1
assert keep >= 1
types = ['segmenter']
regions = RegionNames
models = ['clinical', 'transfer']
folds = list(range(5))
n_trains = [5, 10, 20, 50, 100, 200, None]
runs = []
for model in models:
    for fold in folds:
        for n_train in n_trains:
            runs.append(f'{model}-fold-{fold}-samples-{n_train}')

for type in types:
    for region in regions:
        for run in runs:
            # Print model.
            id = f'{type}:{region}:{run}'
            print(f'model - {id}')

            # Get checkpoints.
            run_folder = os.path.join(config.directories.models, f'{type}-{region}', run)
            if not os.path.exists(run_folder):
                continue
            del_ckpts = list(sorted(os.listdir(run_folder)))

            # Remove 'last' checkpoint.
            if 'last.ckpt' in del_ckpts:
                del_ckpts.remove('last.ckpt')

            # Remove 'best' checkpoints.
            rem = len(del_ckpts) - keep
            del_ckpts = del_ckpts[:rem]

            # Get full paths.
            del_ckpts = [os.path.join(run_folder, f) for f in del_ckpts]

            if dry_run:
                for ckpt in del_ckpts:
                    print(f'\tDELETE - {ckpt}')
            else:
                for ckpt in del_ckpts:
                    print(f'\tDELETE - {ckpt}')
                    os.remove(ckpt)
