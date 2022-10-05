import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD
from typing import Dict, List, Optional, Tuple
import wandb

from mymi import config
from mymi.losses import DiceLoss
from mymi.metrics import batch_mean_dice, batch_mean_all_distances
from mymi import types

from ..networks import UNet2D

class Segmenter2D(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module = DiceLoss(),
        metrics: List[str] = [],
        predict_logits: bool = False):
        super().__init__()
        self._loss = loss
        self._log_args = {
            'on_epoch': True,
            'on_step': False,
        }
        self._max_image_batches = 30
        self._metrics = metrics
        self._network = UNet2D()
        self._predict_logits = predict_logits
        self.save_hyperparameters()

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str,
        **kwargs: Dict) -> pl.LightningModule:
        # Load model.
        model_name, run_name, checkpoint = Segmenter2D.replace_checkpoint_aliases(model_name, run_name, checkpoint)
        filepath = os.path.join(config.directories.models, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Checkpoint '{checkpoint}' not found for auto-encoder run '{model_name}:{run_name}'.")
        return Segmenter2D.load_from_checkpoint(filepath, **kwargs)

    @staticmethod
    def replace_checkpoint_aliases(
        model_name: str,
        run_name: str,
        checkpoint: str) -> Tuple[str, str, str]:
        # Find best checkpoint.
        if checkpoint == 'BEST': 
            dirpath = os.path.join(config.directories.models, model_name, run_name)
            if not os.path.exists(dirpath):
                raise ValueError(f"Run '{run_name}' not found for segmenter-2D '{model_name}'.")
            checkpoint = list(sorted(os.listdir(dirpath)))[-1].replace('.ckpt', '')
        return (model_name, run_name, checkpoint)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        if self._predict_logits:
            pred = pred.cpu().numpy()
            return pred

        # Apply thresholding.
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = pred.cpu().numpy().astype(np.bool)

        return pred

    def training_step(self, batch, _):
        # Forward pass.
        _, x = batch
        y = x.squeeze(1).type(torch.bool)       # Remove 'channel' dimension from label.
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(np.bool)
        self.log('train/loss', loss, **self._log_args)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('train/dice', dice, **self._log_args)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        descs, x = batch
        y = x.squeeze(1).type(torch.bool)       # Remove 'channel' dimension from label.
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('val/loss', loss, **self._log_args, sync_dist=True)
        self.log(f"val/batch/loss/{descs[0]}", loss, on_epoch=False, on_step=True)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('val/dice', dice, **self._log_args, sync_dist=True)
            self.log(f"val/batch/dice/{descs[0]}", dice, on_epoch=False, on_step=True)

        # Log predictions.
        if self.logger:
            class_labels = {
                1: 'foreground'
            }
            for i, desc in enumerate(descs):
                if batch_idx < self._max_image_batches:
                    # Get images.
                    x_img, y_img, y_hat_img = x[i, 0].cpu().numpy(), y[i], y_hat[i]

                    # Transpose to image co-ordinates.
                    x_img = np.transpose(x_img)
                    y_img = np.transpose(y_img)
                    y_hat_img = np.transpose(y_hat_img)

                    # Send image.
                    image = wandb.Image(
                        x_img,
                        caption=desc,
                        masks={
                            'ground_truth': {
                                'mask_data': y_img,
                                'class_labels': class_labels
                            },
                            'predictions': {
                                'mask_data': y_hat_img,
                                'class_labels': class_labels
                            }
                        }
                    )
                    self.logger.experiment.log({ desc: image })
