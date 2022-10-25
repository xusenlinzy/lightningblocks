import argparse
import os
from itertools import chain
from typing import List, Union
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from lightningnlp.task.uie.model import UIE
from lightningnlp.callbacks import Logger


logger = Logger("UIE")


def validate_onnx(tokenizer: PreTrainedTokenizerBase, pt_model: PreTrainedModel, onnx_path: Union[Path, str], strict: bool = True, atol: float = 1e-05):
    from onnxruntime import InferenceSession, SessionOptions
    logger.info("Validating ONNX model...")
    if strict:
        ref_inputs = tokenizer('装备', "印媒所称的“印度第一艘国产航母”—“维克兰特”号", add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")

    else:
        batch_size = 2
        seq_length = 6
        dummy_input = ([(" ".join([tokenizer.unk_token]) * seq_length)] * batch_size)
        ref_inputs = dict(tokenizer(dummy_input, return_tensors="pt"))
    ref_outputs = pt_model(**ref_inputs)
    ref_outputs_dict = {}
    for name, value in ref_outputs.items():
        if name == "past_key_values":
            name = "present"
        ref_outputs_dict[name] = value
    options = SessionOptions()
    session = InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])

    onnx_inputs = {name: value.numpy() for name, value in ref_inputs.items()}
    onnx_named_outputs = ['start_prob', 'end_prob']
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)

    if not onnx_outputs_set.issubset(ref_outputs_set):
        logger.info(f"\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}")

        raise ValueError(f"Outputs doesn't match between reference model and ONNX exported model: {onnx_outputs_set.difference(ref_outputs_set)}")

    else:
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})")

    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        ref_value = ref_outputs_dict[name].detach().numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')
        if ort_value.shape != ref_value.shape:
            logger.info(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")

            raise ValueError(f"Outputs shape doesn't match between reference model and ONNX exported model: Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)")

        else:
            logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")
        if not np.allclose(ref_value, ort_value, atol=atol):
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(f"Outputs values doesn't match between reference model and ONNX exported model: Got max "
                             f"absolute difference of: {np.amax(np.abs(ref_value - ort_value))}")

        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")


def export_onnx(args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, device: torch.device, input_names: List[str], output_names: List[str]):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        model.config.return_dict = True
        model.config.use_cache = False
        if not args.output_path.exists():
            args.output_path.mkdir(parents=True)
        save_path = args.output_path / "inference.onnx"
        dynamic_axes = {name: {0: 'batch', 1: 'sequence'} for name in chain(input_names, output_names)}

        batch_size = 2
        seq_length = 6
        dummy_input = ([(" ".join([tokenizer.unk_token]) * seq_length)] * batch_size)
        inputs = dict(tokenizer(dummy_input, return_tensors="pt"))
        if save_path.exists():
            logger.warning(f'Overwrite model {save_path.as_posix()}')
            save_path.unlink()
        torch.onnx.export(model, (inputs,), save_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, do_constant_folding=True, opset_version=11)

    if not os.path.exists(save_path):
        logger.error('Export Failed!')
    return save_path


def export_onnx_model(args):
    if args.output_path is None:
        args.output_path = args.model_path

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = UIE.from_pretrained(args.model_path)
    device = torch.device('cpu')
    input_names = [
        'input_ids',
        'token_type_ids',
        'attention_mask',
    ]
    output_names = [
        'start_prob',
        'end_prob'
    ]

    logger.info("Export Tokenizer Config...")

    export_tokenizer(args)

    logger.info("Export ONNX Model...")

    save_path = export_onnx(
        args, tokenizer, model, device, input_names, output_names)
    validate_onnx(tokenizer, model, save_path)

    logger.info(f"All good, model saved at: {save_path.as_posix()}")


def export_tokenizer(args):
    for tokenizer_fine in ['tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt']:
        file_from = args.model_path / tokenizer_fine
        file_to = args.output_path/tokenizer_fine
        if file_from.resolve() == file_to.resolve():
            continue
        shutil.copyfile(file_from, file_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=Path, required=True,
                        default='./checkpoint/model_best', help="The path to model parameters to be loaded.")
    parser.add_argument("-o", "--output_path", type=Path, default=None,
                        help="The path of model parameter in static graph to be saved.")
    args = parser.parse_args()
    export_onnx_model(args)
