import hashlib
import math
import os
import re
import warnings

import numpy as np
import unicodedata


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if not seps or len(text) <= maxlen:
        return [text]
    pieces = text.split(seps[0])
    text, texts = '', []
    for i, p in enumerate(pieces):
        if text and p and len(text) + len(p) > maxlen - 1:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
            text = ''
        text = text + p if i + 1 == len(pieces) else text + p + seps[0]
    if text:
        texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
    return texts


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def auto_splitter(input_texts, max_text_len, split_sentence=False):
    """
    Split the raw texts automatically for model inference.
    Args:
        input_texts (List[str]): input raw texts.
        max_text_len (int): cutting length.
        split_sentence (bool): If True, sentence-level split will be performed.
    return:
        short_input_texts (List[str]): the short input texts for model inference.
        input_mapping (dict): mapping between raw text and short input texts.
    """
    input_mapping = {}
    short_input_texts = []
    cnt_short = 0
    for cnt_org, text in enumerate(input_texts):
        sens = cut_chinese_sent(text) if split_sentence else [text]
        for sen in sens:
            lens = len(sen)
            if lens <= max_text_len:
                short_input_texts.append(sen)
                if cnt_org in input_mapping:
                    input_mapping[cnt_org].append(cnt_short)
                else:
                    input_mapping[cnt_org] = [cnt_short]
                cnt_short += 1
            else:
                temp_text_list = [sen[i: i + max_text_len] for i in range(0, lens, max_text_len)]

                short_input_texts.extend(temp_text_list)
                short_idx = cnt_short
                cnt_short += math.ceil(lens / max_text_len)
                temp_text_id = [short_idx + i for i in range(cnt_short - short_idx)]
                if cnt_org in input_mapping:
                    input_mapping[cnt_org].extend(temp_text_id)
                else:
                    input_mapping[cnt_org] = temp_text_id
    return short_input_texts, input_mapping


def lowercase_and_normalize(text):
    """转小写，并进行简单的标准化
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


def check_object_type(object, check_type, name):
    if not isinstance(object, check_type):
        raise TypeError(f"The type of {name} must be {check_type}, but got {type(object)}.")


def md5file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
