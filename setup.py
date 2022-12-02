from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


install_requires = [
    "colorama",
    "colorlog",
    "datasets",
    "faster_tokenizer",
    "jieba",
    "onnx",
    "onnxconverter_common",
    "onnxruntime",
    "pypinyin",
    "pytorch_lightning",
    "scikit_learn",
    "scipy",
    "seaborn",
    "sentencepiece",
    "setuptools",
    "spacy",
    "streamlit",
    "torchmetrics",
    "transformers",
]


setup(
    name="lightningnlp",
    version="1.0.0",
    description="Pytorch-lightning Code Blocks for NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT Licence",
    url="https://github.com/xusenlinzy/lightningblocks",
    author="xusenlin",
    author_email="1659821119@qq.com",
    ikeywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.7",
    setup_requires=[],
    packages=find_packages(),
    install_requires=install_requires,
)
