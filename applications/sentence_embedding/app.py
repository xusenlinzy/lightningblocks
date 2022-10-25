import torch
from scipy.spatial.distance import cosine
from lightningnlp.models import RoFormerModel
from transformers import BertTokenizer
import gradio as gr


model_name_or_path = "/home/xusenlin/nlp/LightningNLP/pretrained_models/roformer_chinese_sim_char_ft_small"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = RoFormerModel.from_pretrained(model_name_or_path, add_pooling_layer=True)


def sim(text1, text2, text3):
    # Tokenize input texts
    texts = [
        text1,
        text2,
        text3,
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
    return cosine_sim_0_1, cosine_sim_0_2


title = "Sentence Embeddings"
examples = [
    ["今天天气不错",
    "今天天气很好",
    "今天天气不好"],
    ["给我推荐一款红色的车",
    "给我推荐一款黑色的车",
    "麻烦来一辆红车"],
]

demo = gr.Interface(
    sim,
    [
        gr.Textbox(lines=5, label="Input Text One"),
        gr.Textbox(lines=5, label="Input Text Two"),
        gr.Textbox(lines=5, label="Input Text Three"),
    ],
    [
        gr.Number(label="Cosine similarity between text one and two"),
        gr.Number(label="Cosine similarity between text one and three"),
    ],
    title=title,
    examples=examples,
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
