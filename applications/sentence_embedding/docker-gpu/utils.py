import torch
import numpy as np
from typing import List, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class SentenceEmbeddings(object):
    """
    A class for embedding sentences.
    """

    def __init__(
        self,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        pooler: str = None,
    ):

        self.tokenizer = tokenizer
        self.pooler = pooler if pooler is not None else "cls"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        sentence: Union[str, List[str]],
        normalize_to_unit: bool = True,
        keepdim: bool = False,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> np.ndarray:

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
        for batch_id in range(total_batch):
            inputs = self.tokenizer(
                sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)

            if self.pooler == "cls":
                embeddings = outputs.pooler_output
            elif self.pooler == "cls_before_pooler":
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                raise NotImplementedError
            if normalize_to_unit:
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embedding_list.append(embeddings.cpu())

        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        return embeddings.numpy()
