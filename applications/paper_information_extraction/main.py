import uvicorn
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from lightningblocks.task.uie import UIEPredictor


# 应用实例化
app = FastAPI()

schema = ["疾病", "症状、临床表现", "身体物质和身体部位", "药物", "检查、治疗或预防程序", "部门科室",
          "医学检验项目", "微生物", "检查设备和治疗设备", "人名、虚构的人物形象", "公司、集团", "游戏",
          "篮球队、足球队、社团、帮派等", "电影", "国家、城市、乡镇、大洲等", "职称、头衔",
          "中央行政机关和地方行政机关", "旅游景点", "小说、杂志等书籍"]

MODEL_CLASSES = {
    "uie": ("uie_base_pytorch", None, ["时间", "地点", "人物"]),
    "uie-paper": ("checkpoint/uie_paper", None, schema + ["专有名词"]),
}


# 导入模型
@st.cache(hash_funcs={UIEPredictor: id})
def load_uie(model_name_or_path, schema, engine="pytorch", device="cpu") -> UIEPredictor:
    return UIEPredictor(model_name_or_path=model_name_or_path, schema=schema, engine=engine, device=device)


# 定义数据格式
class Data(BaseModel):
    uie_schema: str
    input: Union[str, List[str]]  # 可输入一个句子或多个句子
    model_name: str = "uie"
    position_prob: float = 0.5
    max_seq_len: int = 512
    split_sentence: bool = False
    device: str = "cpu"
    engine: str = "pytorch"


# uie接口
@app.post('/uie')
def uie(data: Data):
    engine = getattr(data, "engine", "pytorch")

    model_name = getattr(data, "model_name", "uie")
    model_path_torch, model_path_onnx, schema = MODEL_CLASSES[model_name.lower()]
    engine = "pytorch" if (engine == "pytorch" or not model_path_onnx) else "onnx"
    model_name_or_path = model_path_torch if engine == "pytorch" else model_path_onnx

    ie = load_uie(model_name_or_path, schema, engine, data.device)
    schema, text = data.uie_schema, data.input
    if schema != "":
        schema = schema.split(" ")
        ie.set_schema(list(set(schema)))

    ie.threshold = data.position_prob
    ie.seqlen = data.max_seq_len
    ie.split = data.split_sentence

    rlt = ie(text)
    res = []
    for r in rlt:
        tmp = {_type: [
            {"text": ent["text"], "start": ent["start"], "end": ent["end"], "probability": float(ent["probability"])}
            for ent in ents] for _type, ents in r.items()}
        res.append(tmp)
    return {'success': True, 'rlt': res[0] if isinstance(text, str) else res}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
