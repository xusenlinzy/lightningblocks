FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install transformers==4.23.1 && \
    pip install numpy && \
    pip install roformer==0.4.3 && \
    pip install uvicorn==0.18.3 && \
    pip install fastapi==0.85.0 && \
    pip cache purge

WORKDIR /app/
RUN mkdir -p /app/model
ADD https://huggingface.co/junnyu/roformer_v2_chinese_char_small/resolve/main/config.json /app/model/
ADD https://huggingface.co/junnyu/roformer_v2_chinese_char_small/resolve/main/pytorch_model.bin /app/model/
ADD https://huggingface.co/junnyu/roformer_v2_chinese_char_small/resolve/main/vocab.txt /app/model/
ADD https://huggingface.co/junnyu/roformer_v2_chinese_char_small/resolve/main/tokenizer_config.json /app/model/

COPY . /app
CMD ["python", "app.py"]
