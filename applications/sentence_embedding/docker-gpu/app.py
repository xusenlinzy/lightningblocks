import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
from roformer import RoFormerModel
from transformers import BertTokenizerFast
from utils import SentenceEmbeddings


# 应用实例化
app = FastAPI()

model_path = "/app/model"
tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)
model = RoFormerModel.from_pretrained(model_path, add_pooling_layer=True)
pipeline = SentenceEmbeddings(model, tokenizer)


# 定义数据格式
class Data(BaseModel):
    input: Union[str, List[str]]  # 可输入一个句子或多个句子
    max_seq_len: int = 256
    batch_size: int = 64


# encode接口
@app.post('/encode')
def encode(data: Data):
    vecs = pipeline.encode(data.input, max_length=data.max_seq_len, batch_size=data.batch_size).tolist()
    return {'success': True, 'vecs': vecs}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4000)
