import time
import gradio as gr
import streamlit as st
from lightningblocks.utils.app import make_color_palette, visualize_ner
from lightningblocks.task.uie import UIEPredictor

MODEL_CLASSES = {
    "uie": "./uie_base_pytorch",
    "uie-paper": "./checkpoint/uie-paper",
}


@st.cache(hash_funcs={UIEPredictor: id})
def load_ie(model_name_or_path, schema, prob, max_seq_len, device="cpu", split_sentence=False):
    return UIEPredictor(model_name_or_path=model_name_or_path, schema=schema, position_prob=prob,
                        max_seq_len=max_seq_len, device=device, split_sentence=split_sentence)


def extract(text, schema, schema_options, model_name, max_seq_len, device, split_sentence, use_fp16):
    schema = (f"{schema.strip()} " + " ".join(schema_options)).strip() if schema_options is not None else schema.strip()
    schema = schema.split(" ")
    pipeline = UIEPredictor(model_name_or_path=MODEL_CLASSES[model_name], schema=schema, device=device,
                            max_seq_len=max_seq_len, split_sentence=split_sentence, use_fp16=use_fp16)

    start = time.time()
    rlt = pipeline(text)
    running_time = time.time() - start
    colors = make_color_palette(schema)

    html = visualize_ner(text, rlt[0], colors)
    html = f"{html}"

    res = []
    for r in rlt:
        tmp = {_type: [
            {"text": ent["text"], "start": ent["start"], "end": ent["end"], "probability": float(ent["probability"])}
            for ent in ents] for _type, ents in r.items()}
        res.append(tmp)

    return running_time, res, html


demo = gr.Interface(
    extract,
    [
        gr.Textbox(placeholder="Enter sentence here...", lines=5),
        gr.Textbox(placeholder="Enter schema here...", lines=2),
        gr.CheckboxGroup(
            ["疾病", "症状、临床表现", "身体物质和身体部位", "药物", "检查、治疗或预防程序", "部门科室",
             "医学检验项目", "微生物", "检查设备和治疗设备", "人名、虚构的人物形象", "公司、集团", "游戏",
             "篮球队、足球队、社团、帮派等", "电影", "国家、城市、乡镇、大洲等", "职称、头衔",
             "中央行政机关和地方行政机关", "旅游景点", "小说、杂志等书籍", "专有名词"],
        ),
        gr.Radio(["uie", "uie-paper"], value="uie"),
        gr.Slider(0, 512, value=256),
        gr.Radio(["cpu", "gpu"], value="cpu"),
        gr.Checkbox(label="smart split sentence?"),
        gr.Checkbox(label="use fp16 speed strategy?"),
    ],
    [gr.Number(label="Run Time"), gr.Json(label="Result"), gr.HTML(label="Visualize")],
    examples=[
        ["可伴发血肿或脑梗死而出现局灶性神经体征，如肢体瘫痪及颅神经异常等。"],
        ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"],
        ["实现社会主义现代化,重要的是实现人的现代化。人的问题是当代哲学研究的重要课题。袁贵仁同志主编的《人的哲学》一书,"
         "运用马克思主义的基本理论,对人的问题作了比较全面深入的理论探索。提出了不少新颖独到的见解和探索人的哲学的新思路。"],
    ],
    title="Named Entity Recognition by UIE Model",
    description=
    """     
    ✨   UIE(Universal Information Extraction)]：Yaojie Lu等人在`ACL-2022`中提出了通用信息抽取统一框架`UIE`。
    ✨   该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。
    ✨   为了方便大家使用UIE的强大能力，`PaddleNLP`借鉴该论文的方法，基于`ERNIE 3.0`知识增强预训练模型，训练并开源了首个中文通用信息抽取模型`UIE`。
    ✨   该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。
    """,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
