import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from lightningnlp.callbacks import Logger, tqdm
from lightningnlp.task.uie.model import UIE, UIEM
from lightningnlp.task.uie.utils import SpanEvaluator, IEDataset, IEMapDataset
from lightningnlp.task.uie.utils import unify_prompt_name, get_relation_type_dict

logger = Logger("UIE")


@torch.no_grad()
def evaluate(model, metric, data_loader, device='gpu', loss_fn=None, show_bar=True, multilingual=False):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`torch.nn.Module`): A model to classify texts.
        metric(obj:`Metric`): The evaluation metric.
        data_loader(obj:`torch.utils.data.DataLoader`): The dataset loader which generates batches.
        multilingual(bool): Whether is the multilingual model.
    """
    return_loss = False
    if loss_fn is not None:
        return_loss = True

    model.eval()
    metric.reset()

    loss_list = []
    loss_sum = 0
    loss_num = 0

    if show_bar:
        data_loader = tqdm(data_loader, desc="Evaluating", unit='batch')

    for batch in data_loader:
        if multilingual:
            input_ids, position_ids, att_mask, start_ids, end_ids = batch
            if device == "gpu":
                input_ids = input_ids.cuda()
                att_mask = att_mask.cuda()
                position_ids = position_ids.cuda()

            outputs = model(
                input_ids=input_ids, position_ids=position_ids, attention_mask=att_mask,
            )
        else:
            input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
            if device == "gpu":
                input_ids = input_ids.cuda()
                att_mask = att_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            outputs = model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask,
            )

        start_prob, end_prob = outputs[0], outputs[1]
        if device == 'gpu':
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()

        start_ids = start_ids.type(torch.float32)
        end_ids = end_ids.type(torch.float32)

        if return_loss:
            # Calculate loss
            loss_start = loss_fn(start_prob, start_ids)
            loss_end = loss_fn(end_prob, end_ids)

            loss = (loss_start + loss_end) / 2.0
            loss = float(loss)
            loss_list.append(loss)

            loss_sum += loss
            loss_num += 1

            if show_bar:
                data_loader.set_postfix(
                    {
                        'dev loss': f'{loss_sum / loss_num:.5f}'
                    }
                )

        # Calcalate metric
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)

    precision, recall, f1 = metric.accumulate()
    model.train()

    if return_loss:
        loss_avg = sum(loss_list) / len(loss_list)
        return loss_avg, precision, recall, f1
    else:
        return precision, recall, f1


def do_eval(args):
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    if args.multilingual:
        model = UIEM.from_pretrained(args.model_path)
    else:
        model = UIE.from_pretrained(args.model_path)

    if args.device == "gpu":
        model = model.cuda()

    test_ds = IEDataset(
        args.test_path, tokenizer=tokenizer, max_seq_len=args.max_seq_len, multilingual=args.multilingual
    )

    class_dict = {}
    relation_data = []
    relation_type_dict = None
    if args.debug:
        for data in test_ds.dataset:
            class_name = unify_prompt_name(data['prompt'])
            # Only positive examples are evaluated in debug mode
            if len(data['result_list']) != 0:
                p = "的" if args.schema_lang == "ch" else " of "
                if p not in data['prompt']:
                    class_dict.setdefault(class_name, []).append(data)
                else:
                    relation_data.append((data['prompt'], data))
        relation_type_dict = get_relation_type_dict(relation_data, schema_lang=args.schema_lang)
    else:
        class_dict["all_classes"] = test_ds

    for key in class_dict.keys():
        if args.debug:
            test_ds = IEMapDataset(
                class_dict[key], tokenizer=tokenizer, max_seq_len=args.max_seq_len, multilingual=args.multilingual
            )
        else:
            test_ds = class_dict[key]

        test_data_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        metric = SpanEvaluator()
        precision, recall, f1 = evaluate(model, metric, test_data_loader, args.device)

        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))

    if args.debug and relation_type_dict is not None:
        for key in relation_type_dict.keys():
            test_ds = IEMapDataset(
                relation_type_dict[key], tokenizer=tokenizer, max_seq_len=args.max_seq_le, multilingual=args.multilingual
            )
            test_data_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

            metric = SpanEvaluator()
            precision, recall, f1 = evaluate(model, metric, test_data_loader, args.device)

            logger.info("-----------------------------")
            logger.info("Class Name: X的%s" % key)
            logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))


if __name__ == "__main__":
    import argparse

    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None,
                        help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None,
                        help="The path of test set.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--debug", action='store_true',
                        help="Precision, recall and F1 score are calculated for each class separately if this option is enabled.")
    parser.add_argument("--multilingual", action='store_true',
                        help="Whether is the multilingual model.")
    parser.add_argument("--schema_lang", choices=["ch", "en"], default="ch",
                        help="Select the language type for schema.")

    args = parser.parse_args()
    # yapf: enable

    do_eval(args)
