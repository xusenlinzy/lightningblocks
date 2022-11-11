from typing import List, Union, Dict, Any

import numpy as np
import torch

from lightningnlp.callbacks import Logger, tqdm
from lightningnlp.core import PredictorBase

logger = Logger("Text Classification")


class TextClassificationPredictor(PredictorBase):
    """
    A class for Text Classification task predictor.
    """
    def __init__(self, *args,  label_map: Dict[int, Any] = None, **kwargs):
        self.label_map = label_map
        super().__init__(*args, **kwargs)
        if label_map is not None:
            self.model.config.label_map = label_map

    @torch.no_grad()
    def predict(
        self,
        text_a: Union[str, List[str]],
        text_b: Union[str, List[str]] = None,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> Union[dict, List[dict]]:
        
        single_sentence = False
        if isinstance(text_a, str):
            text_a = [text_a]
            if text_b is not None and isinstance(text_b, str):
                text_b = [text_b]
            single_sentence = True

        output_list = []
        total_batch = len(text_a) // batch_size + (1 if len(text_a) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_text_a = text_a[batch_id * batch_size: (batch_id + 1) * batch_size]
            if text_b is not None:
                batch_text_b = text_b[batch_id * batch_size: (batch_id + 1) * batch_size]
                inputs = self.tokenizer(
                    batch_text_a,
                    batch_text_b,
                    max_length=max_length,
                    padding=True,
                    truncation='only_second',
                    return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(
                    batch_text_a,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
        
            inputs = self._prepare_inputs(inputs)
            outputs = self.model(**inputs)

            outputs = np.asarray(outputs['logits']).argmax(-1)
            output_list.extend(outputs)

        if hasattr(self.model.config, "label_map"):
            output_list = [self.model.config.label_map[o] for o in output_list]

        return output_list[0] if single_sentence else output_list
