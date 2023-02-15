import gradio as gr

from lightningnlp.task.text_classification import TextClassificationPipeline

MODEL_PATH_MAP = {
    "fc": "outputs/tnews/fc/bert-fc",
}


def load_pipeline(model_name, device, max_seq_len, use_fp16=False):
    return TextClassificationPipeline(model_name=model_name,
                                      model_type="bert",
                                      model_name_or_path=MODEL_PATH_MAP[model_name],
                                      device=device,
                                      max_seq_len=max_seq_len,
                                      use_fp16=use_fp16)


def classification(text, model_name, max_seq_len, device, use_fp16):
    # reset pipeline
    pipeline = load_pipeline(model_name, device, max_seq_len, use_fp16)

    return pipeline(text)[0]


demo = gr.Interface(
    classification,
    [
        gr.Textbox(placeholder="Enter sentence here...", lines=5),
        gr.Radio(["fc"], value="fc"),
        gr.Slider(0, 512, value=256),
        gr.Radio(["cpu", "cuda"], value="cpu"),
        gr.Checkbox(label="use fp16 speed strategy?"),
    ],
    "label",
    examples=[
        ["第五季｜北有伊利小说门，南有汉鼎网红门"],
        ["锡林郭勒千里草原风景大道”主题推介会在京举行"],
        ["公婆花80万给我装修新婚房，说是村里最豪华的，你们觉得呢？"],
        ["任务猎先下卡纳莎还是先下王子，你真的想好了吗？"],
    ],
    title="Text Classification for Chinese TNEWS",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7890)
