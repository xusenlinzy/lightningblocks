from collections import defaultdict
from typing import List, Union

from lightningnlp.callbacks import Logger
from lightningnlp.task.named_entity_recognition.auto.predictor import get_auto_ner_predictor
from lightningnlp.task.named_entity_recognition.predictor import set2json
from lightningnlp.utils.common import auto_splitter

logger = Logger("AUTO NER")


class NerPipeline(object):
    def __init__(
            self,
            model_name="crf",
            model_type="bert",
            model=None,
            model_name_or_path=None,
            tokenizer=None,
            device="cpu",
            use_fp16=False,
            max_seq_len=512,
            batch_size=64,
            split_sentence=False,
            schema2prompt=None,
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
        self._split_sentence = split_sentence
        self._schema2prompt = schema2prompt
        self._load_weights = load_weights

        self._prepare_predictor()

    def _prepare_predictor(self):
        logger.info(f">>> [Pytorch InferBackend of {self._model_type}-{self._model_name}] Creating Engine ...")
        self.inference_backend = get_auto_ner_predictor(
            self._model_name,
            self._model_type,
            model=self._model,
            model_name_or_path=self._model_name_or_path,
            tokenizer=self._tokenizer,
            device=self._device,
            use_fp16=self._use_fp16,
            load_weights=self._load_weights,
            schema2prompt=self._schema2prompt,
        )

    def __call__(self, inputs):

        texts = inputs
        if isinstance(texts, str):
            texts = [texts]

        max_prompt_len = len(max(self._schema2prompt.values())) if (
                self._schema2prompt is not None and self._model_name in ["pfn", "grte"]) else -1
        max_predict_len = self._max_seq_len - max_prompt_len - 3

        short_input_texts, self.input_mapping = auto_splitter(
            texts, max_predict_len, split_sentence=self._split_sentence
        )

        results = self.inference_backend.predict(
            short_input_texts, batch_size=self._batch_size, max_length=self._max_seq_len, return_dict=False
        )
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)

        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        for k, vs in input_mapping.items():
            single_results = {}
            offset = 0
            for i, v in enumerate(vs):
                if i == 0:
                    single_results = short_results[v]
                else:
                    for res in short_results[v]:
                        tmp = res[0], res[1] + offset, res[2] + offset, res[3]
                        single_results.add(tmp)
                offset += len(short_inputs[v])
            single_results = set2json(single_results) if single_results else {}
            concat_results.append(single_results)
        return concat_results

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value

    @property
    def split(self):
        return self._split_sentence

    @split.setter
    def split(self, value):
        self._split_sentence = value


def vote(entities_list: List[dict], threshold=0.9) -> dict:
    """
    实体级别的投票方式
    entities_list: 所有模型预测出的一个文件的实体
    threshold: 大于阈值，模型预测出来的实体才能被选中
    """
    counts_dict = defaultdict(int)
    entities = defaultdict(list)

    for _entities in entities_list:
        for _type in _entities:
            for _ent in _entities[_type]:
                counts_dict[(_type, _ent["start"], _ent["end"], _ent["text"])] += 1

    for key in counts_dict:
        if counts_dict[key] >= (len(entities_list) * threshold):
            prob = counts_dict[key] / len(entities_list)
            dic = {"start": key[1], "end": key[2], "text": key[3], "probability": prob}
            entities[key[0]].append(dic)

    return entities


class EnsembleNerPipeline(object):
    """ 基于投票法预测实体
    """

    def __init__(self, predicators: List[NerPipeline]):
        self.predicators = predicators

    def __call__(self, text: Union[str, List[str]], threshold=0.8) -> Union[dict, List[dict]]:
        if isinstance(text, str):
            text = [text]

        all_results = [predicator(text) for predicator in self.predicators]
        return [vote(list(entities_list), threshold=threshold) for entities_list in zip(*all_results)]
