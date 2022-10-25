from typing import List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.optim import Optimizer


class TransformersBaseFinetuning(BaseFinetuning):
    def __init__(self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True):
        r"""`TransformersBaseFinetuning` can be used to create a custom Finetuning Callback.
        Override ``finetune_function`` to put your unfreeze logic.
        Args:
            attr_names: Name(s) of the module attributes of the model to be frozen.
            train_bn: Whether to train Batch Norm layer
        """

        self.attr_names = [attr_names] if isinstance(attr_names, str) else attr_names
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze_using_attr_names(pl_module, self.attr_names, train_bn=self.train_bn)

    def freeze_using_attr_names(self, pl_module, attr_names: List[str], train_bn: bool = True):
        for attr_name in attr_names:
            attr = getattr(pl_module, attr_name, None)
            if attr is None or not isinstance(attr, torch.nn.Module):
                MisconfigurationException(f"Your model must have a {attr} attribute")
            self.freeze(modules=attr, train_bn=train_bn)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        pass
