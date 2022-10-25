import torch
from typing import List, Union, Dict, Set
from lightningblocks.task.named_entity_recognition.predictor import NerPredictor, set2json


class PromptNerPredictor(NerPredictor):

    def __init__(self, schema2prompt: Dict = None, **kwargs):
        self.schema2prompt = schema2prompt
        super().__init__(**kwargs)
        self.model.config.label_list = schema2prompt

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

        return [self.single_sample_predict(sent, max_length, return_dict) for sent in inputs]

    def single_sample_predict(self, inputs: str, max_length: int = 512, return_dict: bool = True):
        infer_inputs = inputs.replace(" ", "-")  # 防止空格导致位置预测偏移

        first_sentences = list(self.schema2prompt.values())
        second_sentences = [infer_inputs] * len(self.schema2prompt)

        batch_inputs = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=max_length,
            padding=True,
            truncation='only_second',
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        batch_inputs['texts'] = [inputs] * len(self.schema2prompt)
        batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

        batch_inputs = self._prepare_inputs(batch_inputs)
        outputs = self.model(**batch_inputs)['predictions']

        return set2json(outputs[0]) if return_dict else outputs[0]
