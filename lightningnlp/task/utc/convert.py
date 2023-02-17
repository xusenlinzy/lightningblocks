import collections
import json
import os
import pickle
import shutil
from base64 import b64decode

import torch

from ...utils.download import get_path_from_url
from ...utils.logger import logger

MODEL_MAP = {
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    "utc-xbase": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/model_state.pdparams",
            "model_config.json":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/config.json",
            "vocab.txt":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/vocab.txt",
            "special_tokens_map.json":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/special_tokens_map.json",
            "tokenizer_config.json":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-xbase/tokenizer_config.json"
        }
    },
    "utc-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/model_state.pdparams",
            "model_config.json":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/config.json",
            "vocab.txt":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/vocab.txt",
            "special_tokens_map.json":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://paddlenlp.bj.bcebos.com/taskflow/zero_shot_text_classification/utc-base/tokenizer_config.json",
        }
    },
}


def build_params_map(model_prefix='encoder', attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        f'{model_prefix}.embeddings.word_embeddings.weight': "encoder.embeddings.word_embeddings.weight",
        f'{model_prefix}.embeddings.position_embeddings.weight': "encoder.embeddings.position_embeddings.weight",
        f'{model_prefix}.embeddings.token_type_embeddings.weight': "encoder.embeddings.token_type_embeddings.weight",
        f'{model_prefix}.embeddings.task_type_embeddings.weight': "encoder.embeddings.task_type_embeddings.weight",
        f'{model_prefix}.embeddings.layer_norm.weight': 'encoder.embeddings.LayerNorm.gamma',
        f'{model_prefix}.embeddings.layer_norm.bias': 'encoder.embeddings.LayerNorm.beta',
    })

    # add attention layers
    for i in range(attention_num):
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.q_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.q_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.k_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.k_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.v_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.v_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.out_proj.weight'] = f'encoder.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.out_proj.bias'] = f'encoder.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm1.weight'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm1.bias'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear1.weight'] = f'encoder.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear1.bias'] = f'encoder.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear2.weight'] = f'encoder.encoder.layer.{i}.output.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear2.bias'] = f'encoder.encoder.layer.{i}.output.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm2.weight'] = f'encoder.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm2.bias'] = f'encoder.encoder.layer.{i}.output.LayerNorm.beta'

    # add pooler
    weight_map.update(
        {
            f'{model_prefix}.pooler.dense.weight': 'encoder.pooler.dense.weight',
            f'{model_prefix}.pooler.dense.bias': 'encoder.pooler.dense.bias',
            'linear_q.weight': 'linear_q.weight',
            'linear_q.bias': 'linear_q.bias',
            'linear_k.weight': 'linear_k.weight',
            'linear_k.bias': 'linear_k.bias',
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        logger.info('=' * 20 + 'save config file' + '=' * 20)

    config = json.load(
        open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))

    if 'init_args' in config:
        config = config['init_args'][0]
    config["architectures"] = ["UIE"]
    config['layer_norm_eps'] = 1e-12
    del config['init_class']

    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'),
              'wt', encoding='utf-8'), indent=4)
    if verbose:
        logger.info('=' * 20 + 'save vocab file' + '=' * 20)

    shutil.copy(os.path.join(input_dir, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))
    special_tokens_map = json.load(open(os.path.join(
        input_dir, 'special_tokens_map.json'), 'rt', encoding='utf-8'))
    json.dump(special_tokens_map, open(os.path.join(output_dir, 'special_tokens_map.json'),
              'wt', encoding='utf-8'))

    tokenizer_config = json.load(
        open(os.path.join(input_dir, 'tokenizer_config.json'), 'rt', encoding='utf-8'))

    if tokenizer_config['tokenizer_class'] == 'ErnieTokenizer':
        tokenizer_config['tokenizer_class'] = "BertTokenizer"
    json.dump(tokenizer_config, open(os.path.join(output_dir, 'tokenizer_config.json'),
              'wt', encoding='utf-8'))
    spm_file = os.path.join(input_dir, 'sentencepiece.bpe.model')

    if os.path.exists(spm_file):
        shutil.copy(spm_file, os.path.join(
            output_dir, 'sentencepiece.bpe.model'))

    if verbose:
        logger.info('=' * 20 + 'extract weights' + '=' * 20)

    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    weight_map.update(build_params_map(
        'ernie', attention_num=config['num_hidden_layers']))

    paddle_paddle_params = pickle.load(
        open(os.path.join(input_dir, 'model_state.pdparams'), 'rb'))
    del paddle_paddle_params['StructuredToParameterName@@']

    for weight_name, weight_value in paddle_paddle_params.items():
        transposed = ''
        if 'weight' in weight_name and ('.encoder' in weight_name or 'pooler' in weight_name or 'linear' in weight_name):
            weight_value = weight_value.transpose()
            transposed = '.T'
        # Fix: embedding error
        if 'word_embeddings.weight' in weight_name:
            weight_value[0, :] = 0
        if weight_name not in weight_map:
            if verbose:
                logger.info(f"{'='*20} [SKIP] {weight_name} {'='*20}")
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        if verbose:
            logger.info(
                f"{weight_name}{transposed} -> {weight_map[weight_name]} {weight_value.shape}")
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def check_model(input_model):
    if os.path.exists(input_model):
        return
    if input_model not in MODEL_MAP:
        raise ValueError('input_model not exists!')

    resource_file_urls = MODEL_MAP[input_model]['resource_file_urls']
    logger.info("Downloading resource files...")

    for key, val in resource_file_urls.items():
        file_path = os.path.join(input_model, key)
        if not os.path.exists(file_path):
            if val.startswith('base64:'):
                base64data = b64decode(val.replace(
                    'base64:', '').encode('utf-8'))
                with open(file_path, 'wb') as f:
                    f.write(base64data)
            else:
                download_path = get_path_from_url(val, input_model)
                if download_path != file_path:
                    shutil.move(download_path, file_path)


def convert_model(input_model_path, output_model_math):
    check_model(input_model_path)
    extract_and_convert(input_model_path, output_model_math, verbose=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", default="utc-base", type=str,
                        help="Directory of input paddle model.\n Will auto download model [utc-base/utc-tiny]")
    parser.add_argument("-o", "--output_model", default="utc_base_pytorch", type=str,
                        help="Directory of output pytorch model")
    args = parser.parse_args()

    convert_model(args.input_model, args.output_model)
