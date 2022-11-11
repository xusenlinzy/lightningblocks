import argparse
import math
import os
import re
from pprint import pprint

import numpy as np
import six
import torch

from lightningnlp.callbacks import Logger, tqdm
from lightningnlp.task.uie.utils import get_bool_ids_greater_than, get_span, get_id_and_prob
from lightningnlp.utils.common import cut_chinese_sent, dbc2sbc

logger = Logger("UIE")


class ONNXInferBackend(object):
    def __init__(self, model_name_or_path, device='cpu', use_fp16=False):
        from onnxruntime import InferenceSession, SessionOptions

        logger.info(">>> [ONNXInferBackend] Creating Engine ...")
        onnx_model = float_onnx_file = os.path.join(model_name_or_path, "inference.onnx")

        if not os.path.exists(onnx_model):
            raise OSError(f'{onnx_model} not exists!')
        infer_model_dir = model_name_or_path

        if device == "gpu":
            providers = ['CUDAExecutionProvider']
            logger.info(">>> [ONNXInferBackend] Use GPU to inference ...")

            if use_fp16:
                logger.info(">>> [ONNXInferBackend] Use FP16 to inference ...")

                from onnxconverter_common import float16
                import onnx

                fp16_model_file = os.path.join(infer_model_dir, "fp16_model.onnx")
                onnx_model = onnx.load_model(float_onnx_file)
                trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model_file)
                onnx_model = fp16_model_file
        else:
            providers = ['CPUExecutionProvider']
            logger.info(">>> [ONNXInferBackend] Use CPU to inference ...")

        sess_options = SessionOptions()
        self.predictor = InferenceSession(onnx_model, sess_options=sess_options, providers=providers)

        if device == "gpu":
            try:
                assert 'CUDAExecutionProvider' in self.predictor.get_providers()
            except AssertionError as e:
                raise AssertionError(f"The environment for GPU inference is not set properly. A possible cause is "
                                     f"that you had installed both onnxruntime and onnxruntime-gpu. Please run the "
                                     f"following commands to reinstall: \n 1) pip uninstall -y onnxruntime "
                                     f"onnxruntime-gpu \n 2) pip install onnxruntime-gpu") from e

        logger.info(">>> [InferBackend] Engine Created ...")

    def infer(self, input_dict: dict):
        return self.predictor.run(None, dict(input_dict))


class PyTorchInferBackend:
    def __init__(self, model_name_or_path, device='cpu', use_fp16=False, multilingual=False):
        from lightningnlp.task.uie.model import UIE, UIEM

        logger.info(">>> [PyTorchInferBackend] Creating Engine ...")

        if multilingual:
            self.model = UIEM.from_pretrained(model_name_or_path)
        else:
            self.model = UIE.from_pretrained(model_name_or_path)

        self.model.eval()
        self.device = device
        if self.device == 'gpu':
            logger.info(">>> [PyTorchInferBackend] Use GPU to inference ...")
            if use_fp16:
                logger.info(
                    ">>> [PyTorchInferBackend] Use FP16 to inference ...")
                self.model = self.model.half()
            self.model = self.model.cuda()
        else:
            logger.info(">>> [PyTorchInferBackend] Use CPU to inference ...")
        logger.info(">>> [PyTorchInferBackend] Engine Created ...")

    @torch.no_grad()
    def infer(self, input_dict):
        import torch

        for input_name, input_value in input_dict.items():
            input_value = torch.LongTensor(input_value)
            if self.device == 'gpu':
                input_value = input_value.cuda()
            input_dict[input_name] = input_value

        outputs = self.model(**input_dict)
        start_prob, end_prob = outputs[0], outputs[1]

        if self.device == 'gpu':
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
        start_prob = start_prob.detach().numpy()
        end_prob = end_prob.detach().numpy()

        return start_prob, end_prob


class UIEPredictor(object):

    keys_to_ignore_on_gpu = ['offset_mapping', 'texts']  # batch不存放在gpu中的变量

    def __init__(
        self,
        model_name_or_path,
        schema,
        schema_lang="zh",
        multilingual=False,
        engine='pytorch',
        device='cpu',
        position_prob=0.5,
        max_seq_len=512,
        batch_size=64,
        split_sentence=False,
        use_fp16=False,
        is_english_model=False
    ) -> None:

        assert isinstance(device, six.string_types), "The type of device must be string."
        assert device in ['cpu', 'gpu'], "The device must be cpu or gpu."

        self._multilingual = multilingual
        if model_name_or_path in ["uie-m-base", "uie-m-large"]:
            self._multilingual = True

        self._is_en = is_english_model
        if model_name_or_path in ["uie-base-en"] or schema_lang == "en":
            self._is_en = True

        self._engine = engine
        self._model_name_or_path = model_name_or_path
        self._device = device
        self._position_prob = position_prob
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._use_fp16 = use_fp16

        self._schema_tree = None
        self.set_schema(schema)

        self._prepare_predictor()

    def _prepare_predictor(self):
        assert self._engine in ['pytorch', 'onnx'], "engine must be pytorch or onnx!"

        if not os.path.exists(self._model_name_or_path):
            input_path = self._model_name_or_path
            self._model_name_or_path = self._model_name_or_path.replace('-', '_') + '_pytorch'
            if not os.path.exists(self._model_name_or_path):
                from lightningnlp.task.uie.convert import check_model, extract_and_convert

                check_model(input_path)
                extract_and_convert(input_path, self._model_name_or_path)

        if self._multilingual:
            from lightningnlp.task.uie.tokenizer import ErnieMTokenizerFast
            self._tokenizer = ErnieMTokenizerFast.from_pretrained(self._model_name_or_path)
        else:
            from transformers import BertTokenizerFast
            self._tokenizer = BertTokenizerFast.from_pretrained(self._model_name_or_path)

        if self._engine == 'pytorch':
            self.inference_backend = PyTorchInferBackend(self._model_name_or_path,
                                                         device=self._device,
                                                         use_fp16=self._use_fp16,
                                                         multilingual=self._multilingual)

        if self._engine == 'onnx':
            if os.path.exists(os.path.join(self._model_name_or_path, "pytorch_model.bin")) and not os.path.exists(
                    os.path.join(self._model_name_or_path, "inference.onnx")):
                from lightningnlp.task.uie.export_model import export_onnx
                from lightningnlp.task.uie.model import UIE, UIEM

                if self._multilingual:
                    model = UIEM.from_pretrained(self._model_name_or_path)
                else:
                    model = UIE.from_pretrained(self._model_name_or_path)

                input_names = [
                    'input_ids',
                    'token_type_ids',
                    'attention_mask',
                ]

                output_names = [
                    'start_prob',
                    'end_prob'
                ]

                logger.info("Converting to the inference model cost a little time.")

                save_path = export_onnx(
                    self._model_name_or_path, self._tokenizer, model, 'cpu', input_names, output_names)
                logger.info(
                    "The inference model save in the path:{}".format(save_path))
                del model

            self.inference_backend = ONNXInferBackend(self._model_name_or_path,
                                                      device=self._device,
                                                      use_fp16=self._use_fp16)

    def set_schema(self, schema):
        if isinstance(schema, (dict, str)):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def __call__(self, inputs):
        texts = inputs
        if isinstance(texts, str):
            texts = [texts]
        return self._multi_stage_predict(texts)

    def _multi_stage_predict(self, datas):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            datas (list): a list of strings
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `datas`
        """
        results = [{} for _ in range(len(datas))]
        if len(datas) < 1 or self._schema_tree is None:
            return results

        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for data in datas:
                    examples.append({"text": data, "prompt": dbc2sbc(node.name)})
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, data in zip(node.prefix, datas):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find(
                                        "[", 1)].strip()
                                    cls_options = re.search(
                                        r'\[.*?\]$', node.name).group()
                                    # Sentiment classification of xxx [positive, negative]
                                    prompt = prompt_prefix + p + " " + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append({
                                "text": data,
                                "prompt": dbc2sbc(prompt)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1

            result_list = self._single_stage_predict(examples) if examples else []
            if not node.parent_relations:
                relations = [[] for _ in range(len(datas))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {node.name: result_list[v[i]]}
                        elif node.name not in relations[k][i]["relations"].keys():
                            relations[k][i]["relations"][node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(result_list[v[i]])

                new_relations = [[] for _ in range(len(datas))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys() and node.name in relations[i][j]["relations"].keys():
                            for k in range(len(relations[i][j]["relations"][node.name])):
                                new_relations[i].append(relations[i][j]["relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(datas))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self._is_en:
                            prefix[k].append(" of " + result_list[idx][i]["text"])
                        else:
                            prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)

        return results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {"text": prompt[start: end], "probability": prob[i]}
                else:
                    result = {"text": text[start: end], "start": start, "end": end, "probability": prob[i]}

                result_list.append(result)
            results.append(result_list)
        return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
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

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3

        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        encoded_inputs = self._tokenizer(
            text=short_texts_prompts,
            text_pair=short_input_texts,
            stride=2,
            truncation=True,
            max_length=self._max_seq_len,
            padding="max_length" if self._multilingual else "longest",
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np")
        offset_maps = encoded_inputs["offset_mapping"]

        start_prob_concat, end_prob_concat = [], []
        batch_iterator = tqdm(range(0, len(short_input_texts), self._batch_size), desc="Predicting", unit='batch')
        for batch_start in batch_iterator:
            if self._multilingual:
                batch = {key: np.array(value[batch_start: batch_start + self._batch_size], dtype="int64") for key, value
                         in encoded_inputs.items() if key in ["input_ids", "attention_mask"]}
                batch["position_ids"] = (np.cumsum(np.ones_like(batch["input_ids"]), axis=1) - np.ones_like(batch["input_ids"])) * batch["attention_mask"]
            else:
                batch = {key: np.array(value[batch_start: batch_start + self._batch_size], dtype="int64") for key, value
                         in encoded_inputs.items() if key not in self.keys_to_ignore_on_gpu}
            outputs = self.inference_backend.infer(batch)

            start_prob, end_prob = outputs[0], outputs[1]
            start_prob_concat.append(start_prob)
            end_prob_concat.append(end_prob)

        start_prob_concat = np.concatenate(start_prob_concat)
        end_prob_concat = np.concatenate(end_prob_concat)

        start_ids_list = get_bool_ids_greater_than(start_prob_concat, limit=self._position_prob, return_prob=True)
        end_ids_list = get_bool_ids_greater_than(end_prob_concat, limit=self._position_prob, return_prob=True)

        input_ids = encoded_inputs['input_ids'].tolist()
        sentence_ids, probs = [], []
        for start_ids, end_ids, ids, offset_map in zip(start_ids_list, end_ids_list, input_ids, offset_maps):
            # for i in reversed(range(len(ids))):
            #     if ids[i] != 0:
            #         ids = ids[:i]
            #         break
            span_list = get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = get_id_and_prob(span_list, offset_map.tolist())
            sentence_ids.append(sentence_id)
            probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if not short_result:
                continue
            elif 'start' not in short_result[0].keys() and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            single_results = []
            if is_cls_task:
                cls_options = {}
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] in cls_options:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][1] += short_results[v][0]['probability']

                    else:
                        cls_options[short_results[v][0]['text']] = [1, short_results[v][0]['probability']]

                if cls_options:
                    cls_res, cls_info = max(cls_options.items(), key=lambda x: x[1])
                    concat_results.append([{'text': cls_res, 'probability': cls_info[1] / cls_info[0]}])

                else:
                    concat_results.append([])
            else:
                offset = 0
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def predict(self, input_data):
        return self._multi_stage_predict(input_data)

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            f"Invalid schema, value for each key:value pairs should be list or string"
                            f"but {type(v)} received")
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(f"Invalid schema, element should be string or dict, but {type(s)} received")

        return schema_tree

    @property
    def threshold(self):
        return self._position_prob

    @threshold.setter
    def threshold(self, value):
        self._position_prob = value

    @property
    def seqlen(self):
        return self._max_seq_len

    @seqlen.setter
    def seqlen(self, value):
        self._max_seq_len = value

    @property
    def split(self):
        return self._split_sentence

    @split.setter
    def split(self, value):
        self._split_sentence = value


class SchemaTree(object):
    """
    Implementation of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instance of SchemaTree."
        self.children.append(node)


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default='uie_base_pytorch',
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "-p",
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probability.", )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help="Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parser.add_argument(
        "-D",
        "--device",
        choices=['cpu', 'gpu'],
        default="cpu",
        help="Select which device to run model, defaults to gpu."
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=['pytorch', 'onnx'],
        default="pytorch",
        help="Select which engine to run model, defaults to pytorch."
    )
    return parser.parse_args()


if __name__ == '__main__':

    schema = ['Time', 'Player', 'Competition', 'Score']
    ie = UIEPredictor("uie-m-base", schema=schema, schema_lang="en")
    pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))

    # schema = ['时间', '选手', '赛事名称']  # Define the schema for entity extraction
    # ie = UIEPredictor("uie-base", schema=schema)
    #
    # texts = ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！",
    #          "北京时间24日，2022年羽毛球世锦赛中男单选手赵俊鹏直落两局击败上届亚军、12号种子斯里坎特，与石宇奇携手晋级16强。"]
    # pprint(ie(texts))
    #
    # schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
    # ie.set_schema(schema)
    # pprint(
    #     ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))
    #
    # # 关系抽取
    # schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
    # ie.set_schema(schema)  # Reset schema
    # pprint(
    #     ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))
    #
    # # 事件抽取
    # schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}
    # ie.set_schema(schema)  # Reset schema
    # ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')
    #
    # # 评论观点抽取
    # schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
    # ie.set_schema(schema)  # Reset schema
    # pprint(ie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"))
    #
    # # 情感倾向分类
    # schema = '情感倾向[正向，负向]'
    # ie.set_schema(schema)
    # pprint(ie('这个产品用起来真的很流畅，我非常喜欢'))
