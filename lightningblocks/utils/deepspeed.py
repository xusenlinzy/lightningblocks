from typing import Any

import pytorch_lightning as pl
import transformers.deepspeed


class ZeRO3Config:
    def __init__(self, pl_module):
        self.config = pl_module.trainer.strategy.config

    def __call__(self, *args, **kwargs) -> Any:
        return self

    def is_zero3(self) -> bool:
        return self.config.get("zero_optimization") and self.config.get("zero_optimization").get("stage") == 3


def enable_transformers_pretrained_deepspeed_sharding(pl_module: "pl.LightningModule") -> None:
    transformers.deepspeed._hf_deepspeed_config_weak_ref = ZeRO3Config(pl_module)
