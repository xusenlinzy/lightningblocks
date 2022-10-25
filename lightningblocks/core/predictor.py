import torch
from typing import Union, Any, Dict
from collections.abc import Mapping
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from lightningblocks.callbacks import Logger

logger = Logger("PredictorBase")


class PredictorBase(object):
    """
    A class for base predictor.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        model_name_or_path: str = None,
        tokenizer: PreTrainedTokenizerBase = None,
        device: str = "cpu",
        use_fp16: bool = False,
        load_weights: bool = True,
    ):
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.load_weights = load_weights
        self.device = device
        self.use_fp16 = use_fp16

        self._prepare_predictor()

    def _prepare_predictor(self):
        if self.load_weights:
            assert self.model_name_or_path is not None, "The `model_name_or_path` should be specified to load weights."
            self.model = self.model.from_pretrained(self.model_name_or_path)

        if self.tokenizer is None:
            from transformers import BertTokenizerFast

            assert self.model_name_or_path is not None, "The `model_name_or_path` should be specified to load tokenizer."
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name_or_path)

        self.model.eval()

        if self.device == 'cuda':
            logger.info(">>> [PyTorchInferBackend] Use GPU to inference ...")
            if self.use_fp16:
                logger.info(
                    ">>> [PyTorchInferBackend] Use FP16 to inference ...")
                self.model = self.model.half()
            self.model = self.model.cuda()
        else:
            logger.info(">>> [PyTorchInferBackend] Use CPU to inference ...")
        logger.info(">>> [PyTorchInferBackend] Engine Created ...")

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty."
            )
        return inputs

    @torch.no_grad()
    def predict(self, text, **kwargs):
        raise NotImplementedError('Method [predict] should be implemented.')
    