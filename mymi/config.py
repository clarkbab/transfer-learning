from collections import namedtuple
import os
import pandas as pd
from typing import Dict, List, Optional

from mymi import logging

class Directories:
    @property
    def cache(self):
        return os.path.join(self.root, 'cache')

    @property
    def models(self):
        return os.path.join(self.root, 'models')

    @property
    def datasets(self):
        return os.path.join(self.root, 'datasets')

    @property
    def files(self):
        return os.path.join(self.root, 'files')
    
    @property
    def evaluations(self):
        return os.path.join(self.root, 'evaluations')

    @property
    def root(self):
        return os.environ['MYMI_DATA']

    @property
    def runs(self):
        return os.path.join(self.root, 'runs')

    @property
    def temp(self):
        return os.path.join(self.root, 'tmp')

    @property
    def tensorboard(self):
        return os.path.join(self.root, 'reports', 'tensorboard')

    @property
    def wandb(self):
        return os.path.join(self.root, 'reports')

class Formatting:
    @property
    def metrics(self):
        return '.6f'

    @property
    def sample_digits(self):
        return 5

directories = Directories()
formatting = Formatting()

def environ(name: str) -> Optional[str]:
    if name in os.environ:
        return os.environ[name]
    else:
        return None
