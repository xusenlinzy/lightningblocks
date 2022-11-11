from functools import lru_cache

from pypinyin import Style, pinyin
from transformers.models.bert import BertTokenizerFast


class ChineseBertTokenizerFast(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinyin_dict = kwargs.get("pinyin_map")
        self.id2pinyin = kwargs.get("id2pinyin")
        self.pinyin2tensor = kwargs.get("pinyin2tensor")
        self.special_tokens_pinyin_ids = [0] * 8

    def custom_input_ids(self, text_or_text_pair, e):
        return {"pinyin_ids": self.get_pinyin_ids(text_or_text_pair, e)}

    # pinyin_ids
    @lru_cache(maxsize=2021)
    def get_pinyin_locs_map(self, text=None):
        if text is None:
            return None
        pinyin_list = pinyin(
            text,
            style=Style.TONE3,
            heteronym=True,
            errors=lambda x: [["not chinese"] for _ in x],
        )
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        return pinyin_locs
