import gradio as gr
from chatbot_graph import ChatBotGraph


handler = ChatBotGraph()


def chat(message, history):
    history = history or []
    history.append((message, handler.chat_main(message)))
    return history, history


chatbot = gr.Chatbot(label="History").style(color_map=("green", "pink"))
demo = gr.Interface(
    chat,
    ["text", "state"],
    [chatbot, "state"],
    allow_flagging="never",
    examples=[
        ["头痛能治疗吗"],
        ["感冒有什么特效药吗"],
        ["感冒有什么忌口的吗"],
    ],
    title="Knowledge Graph Based Question Answering",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
