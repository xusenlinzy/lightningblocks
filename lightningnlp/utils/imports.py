import operator

from pytorch_lightning.utilities import _module_available
from pytorch_lightning.utilities.imports import _compare_version

_BOLTS_AVAILABLE = _module_available("pl_bolts") and _compare_version("pl_bolts", operator.ge, "0.4.0")
_BOLTS_GREATER_EQUAL_0_5_0 = _module_available("pl_bolts") and _compare_version("pl_bolts", operator.ge, "0.5.0")
_WANDB_AVAILABLE = _module_available("wandb")
_ACCELERATE_AVAILABLE = _module_available("accelerate")
