from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
from typing import List, Optional, Tuple, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.training import exists
from mymi.loaders import Loader
from mymi import logging
from mymi.losses import DiceLoss
from mymi.models.systems import Segmenter
from mymi.reporting.loaders import get_loader_manifest
from mymi import types

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_segmenter(
    datasets: Union[str, List[str]],
    region: str,
    model: str,
    run: str,
    loss: str = 'dice',
    n_epochs: int = 200,
    n_folds: Optional[int] = 5,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    pretrained_model: Optional[types.ModelName] = None,    
    p_val: float = 0.2,
    resume: bool = False,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
    model_name = model
    logging.info(f"Training model '({model_name}, {run})' on datasets '{datasets}' with region '{region}' - pretrained model '{pretrained_model}'.")
    # 'libgcc'
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    # Load datasets.
    if type(datasets) == str:
        datasets = [datasets]
        spacing = ds.get(datasets[0], 'training').params['spacing']
    else:
        spacing = ds.get(datasets[0], 'training').params['spacing']
        for dataset in datasets[1:]:
            # Check for consistent spacing.
            new_spacing = ds.get(dataset, 'training').params['spacing']
            if new_spacing != spacing:
                raise ValueError(f'Datasets must have consistent spacing.')

    # Create transforms.
    rotation = (-5, 5)
    translation = (-50, 50)
    scale = (0.8, 1.2)
    transform = RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')

    # Create data loaders.
    loaders = Loader.build_loaders(datasets, region, extract_patch=True, n_folds=n_folds, n_train=n_train, n_workers=n_workers, p_val=p_val, spacing=spacing, test_fold=test_fold, transform=transform)
    train_loader = loaders[0]
    val_loader = loaders[1]

    # Get loss function.
    if loss == 'dice':
        loss_fn = DiceLoss()
    elif loss == 'scdice':
        loss_fn = DiceLoss(weights=[0, 1])

    # Create model.
    metrics = ['dice']
    if pretrained_model:
        pretrained_model = Segmenter.load(*pretrained_model)
    model = Segmenter(
        loss=loss_fn,
        metrics=metrics,
        pretrained_model=pretrained_model,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            # group=f"{model_name}-{run}",
            project=model_name,
            name=run,
            save_dir=config.directories.wandb)
        logger.watch(model)   # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    checks_path = os.path.join(config.directories.models, model_name, run)
    callbacks = [
        # EarlyStopping(
        #     monitor='val/loss',
        #     patience=5),
        ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=checks_path,
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            monitor='val/loss',
            save_last=True,
            save_top_k=1)
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        # Get the checkpoint path.
        resume_run = resume_run if resume_run is not None else run
        logging.info(f'Loading ckpt {model_name}, {resume_run}, {resume_ckpt}')
        ckpt_path = os.path.join(config.directories.models, model_name, resume_run, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path

    # Perform training.
    trainer = Trainer(
        # accelerator='ddp',
        callbacks=callbacks,
        gpus=list(range(n_gpus)),
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        # plugins=DDPPlugin(find_unused_parameters=False),
        precision=16)

    # Save training information.
    man_df = get_loader_manifest(datasets, region, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
    folderpath = os.path.join(config.directories.runs, model_name, run, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
