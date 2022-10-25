import torch
from typing import List, Union, Dict, Set
from lightningnlp.task.named_entity_recognition.predictor import NerPredictor, set2json
from lightningnlp.callbacks import tqdm


class LearNerPredictor(NerPredictor):

    def __init__(self, schema2prompt: Dict = None, **kwargs):
        self.schema2prompt = schema2prompt
        super().__init__(**kwargs)

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
        label_annotations = list(self.schema2prompt.values())
        label_inputs = self.tokenizer(
            label_annotations,
            padding=True,
            truncation=True,
            max_length=64,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        label_inputs = {f"label_{k}": v for k, v in label_inputs.items()}

        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs = self.tokenizer(
                batch_inputs,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            batch_inputs['texts'] = inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

            batch_inputs = {**batch_inputs, **label_inputs}
            batch_inputs = self._prepare_inputs(batch_inputs)

            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]
