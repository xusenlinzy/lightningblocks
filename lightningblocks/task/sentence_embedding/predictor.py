import torch
from tqdm import tqdm
from numpy import ndarray
from torch import Tensor
from typing import List, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from sklearn.metrics.pairwise import cosine_similarity
from lightningblocks.callbacks import Logger


logger = Logger("Sentence Embeddings")


class SentenceEmbeddings(object):
    """
    A class for embedding sentences, calculating similarities.
    """

    def __init__(
        self,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        device: str = None,
        pooler: str = None,
    ):

        self.tokenizer = tokenizer
        self.model = model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.pooler = pooler if pooler is not None else "cls"

    @torch.no_grad()
    def encode(
        self,
        sentence: Union[str, List[str]],
        device: str = None,
        return_numpy: bool = False,
        normalize_to_unit: bool = True,
        keepdim: bool = False,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
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

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def similarity(
        self,
        queries: Union[str, List[str]],
        keys: Union[str, List[str], ndarray],
        device: str = None
    ) -> Union[float, ndarray]:

        query_vecs = self.encode(queries, device=device, return_numpy=True)  # suppose N queries

        key_vecs = keys if isinstance(keys, ndarray) else self.encode(keys, device=device, return_numpy=True)

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])

        return similarities
