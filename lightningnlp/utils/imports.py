import operator

from lightning_utilities.core.imports import compare_version, module_available

_BOLTS_AVAILABLE = module_available("pl_bolts") and compare_version("pl_bolts", operator.ge, "0.4.0")
_BOLTS_GREATER_EQUAL_0_5_0 = module_available("pl_bolts") and compare_version("pl_bolts", operator.ge, "0.5.0")
_WANDB_AVAILABLE = module_available("wandb")
_ACCELERATE_AVAILABLE = module_available("accelerate")
