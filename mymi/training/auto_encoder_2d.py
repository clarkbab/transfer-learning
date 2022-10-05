import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
from typing import List, Optional, Union

from mymi import config
from mymi import dataset as ds
from mymi.loaders import OtherLoader
from mymi.losses import DiceLoss
from mymi import logging
from mymi.models.systems import AutoEncoder2D

def train_auto_encoder_2d(
    model_name: str,
    run_name: str,
    datasets: Union[str, List[str]],
    bottleneck: int,
    loss: str = 'dice',
    n_epochs: int = 200,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_workers: int = 1,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    use_logger: bool = False) -> None:
    logging.info(f"Training model '({model_name}, {run_name})' on datasets '{datasets}'...")

    # Load partitions.
    if isinstance(datasets, str):
        set = ds.get(datasets, 'training')
        spacing = eval(set.params.spacing[0])
        train_parts = set.partition('train')
        val_parts = [set.partition('validation')]
    else:
        set = ds.get(datasets[0], 'training')
        train_parts = []
        val_parts = []
        for d in datasets:
            set = ds.get(d, 'training')
            train_parts.append(set.partition('train'))
            val_parts.append(set.partition('validation'))

    # Create transforms.
    scale = (0.8, 1.2)
    transform = RandomAffine(
        # degrees=rotation,
        scales=scale,
        # translation=translation,
        default_pad_value='minimum')

    # Create data loaders.
    precision = 'half' if n_gpus > 0 else 'single'
    train_loader = OtherLoader.build(train_parts, num_workers=n_workers, precision=precision, transform=transform)
    val_loader = OtherLoader.build(val_parts, num_workers=n_workers, precision=precision, shuffle=False)

    # Get loss function.
    if loss == 'dice':
        loss_fn = DiceLoss()
    elif loss == 'scdice':
        loss_fn = DiceLoss(weights=[0, 1])

    # Create model.
    metrics = ['dice']
    model = AutoEncoder2D(
        bottleneck=bottleneck,
        loss=loss_fn,
        metrics=metrics)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.wandb)
        # logger.watch(model) # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    checks_path = os.path.join(config.directories.models, model_name, run_name)
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
            save_top_k=3)
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        if resume_checkpoint is None:
            raise ValueError(f"Must pass 'resume_checkpoint' when resuming training run.")
        check_path = os.path.join(checks_path, f"{resume_checkpoint}.ckpt")
        opt_kwargs['resume_from_checkpoint'] = check_path
    
    # Perform training.
    if n_gpus > 0:
        gpus = list(range(n_gpus))
        precision = 16
    else:
        gpus = None
        precision = 32

    trainer = Trainer(
        accelerator='ddp',
        callbacks=callbacks,
        gpus=gpus,
        logger=logger,
        max_epochs=n_epochs,
        n_nodes=n_nodes,
        n_sanity_val_steps=0,
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=precision,
        **opt_kwargs)
    trainer.fit(model, train_loader, val_loader)
