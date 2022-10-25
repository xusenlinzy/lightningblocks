import sys

sys.path.append("../..")

import uvicorn
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
from roformer import RoFormerModel
from transformers import BertTokenizer
from lightningblocks.task.sentence_embedding import SentenceEmbeddings


# 应用实例化
app = FastAPI()


MODEL_CLASSES = {
    "base": "/home/xusenlin/nlp/LightningNLP/pretrained_models/roformer_chinese_sim_char_ft_base",
    "small": "/home/xusenlin/nlp/LightningNLP/pretrained_models/roformer_chinese_sim_char_ft_small",
}


@st.cache(hash_funcs={SentenceEmbeddings: id})
def load_pipeline(model_name="base") -> SentenceEmbeddings:
    model_name_or_path = MODEL_CLASSES[model_name.lower()]
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = RoFormerModel.from_pretrained(model_name_or_path, add_pooling_layer=True)
    return SentenceEmbeddings(model, tokenizer)


# 定义数据格式
class Data(BaseModel):
    input: Union[str, List[str]]  # 可输入一个句子或多个句子
    model_name: str = "base"
    max_seq_len: int = 256


# uie接口
@app.post('/encode')
def encode(data: Data):
    pipeline = load_pipeline(data.model_name)
    vecs = pipeline.encode(data.input).tolist()
    return {'success': True, 'vecs': vecs}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
