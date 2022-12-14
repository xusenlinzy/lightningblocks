import itertools
from typing import List, Union, Dict, Set

import numpy as np
import torch

from .data import DIST_TO_IDX, DataCollatorForW2Ner
from ..predictor import NerPredictor, set2json
from ....utils.logger import tqdm


class W2NerPredictor(NerPredictor):

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        infer_inputs = [t.replace(" ", "-") for t in inputs]  # 防止空格导致位置预测偏移

        outputs = []
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        collate_fn = DataCollatorForW2Ner()
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs = [self._process(example, max_length) for example in batch_inputs]

            batch_inputs = collate_fn(batch_inputs)
            batch_inputs = self._prepare_inputs(batch_inputs)

            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]

    def _process(self, text, max_length):
        tokens = [self.tokenizer.tokenize(word) for word in text[:max_length - 2]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _input_ids = self.tokenizer.convert_tokens_to_ids(pieces)
        _input_ids = np.array([self.tokenizer.cls_token_id] + _input_ids + [self.tokenizer.sep_token_id])

        length = len(tokens)
        _pieces2word = np.zeros((length, len(_input_ids)), dtype=np.bool)
        if self.tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        _dist_inputs = np.zeros((length, length), dtype=np.int)
        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
        for i, j in itertools.product(range(length), range(length)):
            _dist_inputs[i, j] = DIST_TO_IDX[-_dist_inputs[i, j]] + 9 if _dist_inputs[i, j] < 0 else DIST_TO_IDX[
                _dist_inputs[i, j]]

        _dist_inputs[_dist_inputs == 0] = 19

        _grid_mask = np.ones((length, length), dtype=np.bool)
        input_keys = ["input_ids", "pieces2word", "dist_inputs", "grid_mask"]

        encoded_inputs = {k: list(v) for k, v in zip(input_keys, [_input_ids, _pieces2word, _dist_inputs, _grid_mask])}
        encoded_inputs["text"] = text

        return encoded_inputs
    