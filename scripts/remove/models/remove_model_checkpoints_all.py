import os
from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config

keep_n = 1
dry_run = True

modelspath = config.directories.models
models = os.listdir(modelspath)
for model in models:
    runspath = os.path.join(modelspath, model)
    runs = os.listdir(runspath)
    for run in runs:
        ckptspath = os.path.join(runspath, run)
        ckpts = os.listdir(ckptspath)

        # Remove 'last.ckpt'
        if 'last.ckpt' in ckpts:
            ckpts.remove('last.ckpt')

        # Sort and highlight those to delete.
        ckpts = sorted(ckpts)
        ckpts_to_delete = ckpts[:-keep_n]

        for ckpt in ckpts_to_delete:
            if dry_run:
                print(f'{model}:{run}:{ckpt}')
            else:
                print(f'{model}:{run}:{ckpt}')
                ckptpath = os.path.join(ckptspath, ckpt)
                os.remove(ckptpath)
