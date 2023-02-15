from .predictor import get_auto_tc_predictor
from ....utils.logger import logger


class TextClassificationPipeline(object):
    def __init__(
        self,
        model_name="fc",
        model_type="bert",
        model=None,
        model_name_or_path=None,
        tokenizer=None,
        device="cpu",
        use_fp16=False,
        max_seq_len=512,
        batch_size=64,
        load_weights=True,
    ) -> None:

        self._model_name = model_name
        self._model_type = model_type
        self._model = model
        self._model_name_or_path = model_name_or_path
        self._tokenizer = tokenizer
        self._device = device
        self._use_fp16 = use_fp16
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._load_weights = load_weights

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_tc_predictor(
            self._model_name,
            self._model_type,
            model=self._model,
            model_name_or_path=self._model_name_or_path,
            tokenizer=self._tokenizer,
            device=self._device,
            use_fp16=self._use_fp16,
            load_weights=self._load_weights,
        )

    def __call__(self, inputs):

        texts = inputs
        if isinstance(texts, str):
            texts = [texts]

        results = self.inference_backend.predict(
            texts, batch_size=self._batch_size, max_length=self._max_seq_len
        )

        return results

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value
