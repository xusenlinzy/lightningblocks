import time
import gradio as gr
import streamlit as st
from lightningnlp.task.relation_extraction import RelationExtractionPipeline


MODEL_PATH_MAP = {
    "casrel": "outputs/duie/casrel/bert-casrel",
    "spn": "outputs/duie/spn/bert-spn",
    "gplinker": "outputs/duie/gplinker/bert-gplinker",
    "tplinker": "outputs/duie/tplinker/bert-tplinker",
    "pfn": "outputs/duie/pfn/bert-pfn",
    "grte": "outputs/duie/grte/bert-grte",
    "prgc": "outputs/duie/prgc/bert-prgc",
}


@st.cache(hash_funcs={RelationExtractionPipeline: id})
def load_pipeline(model_name, device, max_seq_len, split_sentence=False, use_fp16=False):
    return RelationExtractionPipeline(model_name=model_name, model_type="bert",
                                      model_name_or_path=MODEL_PATH_MAP[model_name],
                                      device=device, max_seq_len=max_seq_len,
                                      split_sentence=split_sentence, use_fp16=use_fp16,)


def extract(text, model_name, max_seq_len, device, split_sentence, use_fp16):
    # reset pipeline
    pipeline = load_pipeline(model_name, device, max_seq_len, split_sentence, use_fp16)

    start = time.time()
    res = pipeline(text)
    running_time = time.time() - start

    return running_time, res


demo = gr.Interface(
    extract,
    [
        gr.Textbox(placeholder="Enter sentence here...", lines=5),
        gr.Radio(["casrel", "spn", "gplinker", "tplinker", "pfn", "grte", "prgc"], value="casrel"),
        gr.Slider(0, 512, value=256),
        gr.Radio(["cpu", "cuda"], value="cpu"),
        gr.Checkbox(label="smart split sentence?"),
        gr.Checkbox(label="use fp16 speed strategy?"),
    ],
    [gr.Number(label="Run Time"), gr.Json(label="Result")],
    examples=[
        ["查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"],
        ["周佛海被捕入狱之后，其妻杨淑慧散尽家产请蒋介石枪下留人，于是周佛海从死刑变为无期，不过此人或许作恶多端，改判没多久便病逝于监狱，据悉是心脏病发作"],
        ["《小王子》是由神田武幸导演，松野达也、増冈弘、松尾佳子、たてかべ和也主演，1978年7月上映的电影"],
    ],
    title="Relation Extraction for Chinese DUIE",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
