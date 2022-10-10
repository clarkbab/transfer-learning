import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from mymi.loaders import get_loader_n_train

if __name__ == '__main__':
    fire.Fire(get_loader_n_train)
