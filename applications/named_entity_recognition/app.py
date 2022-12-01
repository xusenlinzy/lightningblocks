import time
import gradio as gr
import streamlit as st
from lightningnlp.utils.app import make_color_palette, visualize_ner
from lightningnlp.task.named_entity_recognition import NerPipeline


MODEL_PATH_MAP = {
    "crf": "outputs/cmeee/crf/bert-crf",
    "span": "outputs/cmeee/span/bert-span",
    "global-pointer": "outputs/cmeee/global-pointer/bert-global-pointer",
    "tplinker": "outputs/cmeee/tplinker/bert-tplinker",
    "mrc": "outputs/cmeee/mrc/bert-mrc",
    "lear": "outputs/cmeee/lear/bert-lear",
    "w2ner": "outputs/cmeee/w2ner/bert-w2ner",
}


@st.cache(hash_funcs={NerPipeline: id})
def load_pipeline(model_name, device, max_seq_len, split_sentence=False, use_fp16=False):
    return NerPipeline(model_name=model_name, model_type="bert",
                       model_name_or_path=MODEL_PATH_MAP[model_name],
                       device=device, max_seq_len=max_seq_len,
                       split_sentence=split_sentence, use_fp16=use_fp16,)


def extract(text, model_name, max_seq_len, device, split_sentence, use_fp16):
    # reset pipeline
    pipeline = load_pipeline(model_name, device, max_seq_len, split_sentence, use_fp16)

    start = time.time()
    res = pipeline(text)
    running_time = time.time() - start

    labels = list(pipeline.inference_backend.model.config.id2label.values())
    colors = make_color_palette(labels)
    html = visualize_ner(text, res[0], colors)
    html = (
        ""
        + html
        + ""
    )

    return running_time, res, html


demo = gr.Interface(
    extract,
    [
        gr.Textbox(placeholder="Enter sentence here...", lines=5),
        gr.Radio(["crf", "span", "global-pointer", "tplinker", "lear", "w2ner"], value="crf"),
        gr.Slider(0, 512, value=256),
        gr.Radio(["cpu", "cuda"], value="cpu"),
        gr.Checkbox(label="smart split sentence?"),
        gr.Checkbox(label="use fp16 speed strategy?"),
    ],
    [gr.Number(label="Run Time"), gr.Json(label="Result"), gr.HTML(label="Visualize")],
    examples=[
        ["可伴发血肿或脑梗死而出现局灶性神经体征，如肢体瘫痪及颅神经异常等。"],
        ["房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。"],
        ["如非肺炎病例，宜用宽大胶布条紧缠患部以减少其呼吸动作或给镇咳剂抑制咳嗽。"],
    ],
    title="Named Entity Recognition for Chinese Medical Corpus",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
