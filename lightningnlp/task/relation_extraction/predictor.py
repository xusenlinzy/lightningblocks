import torch
from collections import defaultdict
from typing import List, Union, Dict, Set
from lightningnlp.core import PredictorBase
from lightningnlp.callbacks import tqdm


def set2json(labels: Set) -> Dict:
    """ 将三元组集合根据关系类型转换为字典
    """
    res = defaultdict(list)
    for _type, _sub, _obj in labels:
        dic = {"subject": _sub, "object": _obj}
        res[_type].append(dic)
    return res


class RelationExtractionPredictor(PredictorBase):

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 64,
        max_length: int = 512,
        return_dict: bool = True,
    ) -> Union[List[Set], List[Dict]]:

        if isinstance(inputs, str):
            inputs = [inputs]

        infer_inputs = [t.replace(" ", "-") for t in inputs]  # 防止空格导致位置预测偏移

        outputs = []
        total_batch = len(infer_inputs) // batch_size + (1 if len(infer_inputs) % batch_size > 0 else 0)
        for batch_id in tqdm(range(total_batch), desc="Predicting"):
            batch_inputs = infer_inputs[batch_id * batch_size: (batch_id + 1) * batch_size]

            batch_inputs = self.tokenizer(
                batch_inputs,
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )

            batch_inputs['texts'] = inputs[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_inputs["offset_mapping"] = batch_inputs["offset_mapping"].tolist()

            batch_inputs = self._prepare_inputs(batch_inputs)
            batch_outputs = self.model(**batch_inputs)
            outputs.extend(batch_outputs['predictions'])

        return outputs if not return_dict else [set2json(o) for o in outputs]
