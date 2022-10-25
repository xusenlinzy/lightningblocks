import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from lightningblocks.callbacks import Logger, tqdm
from lightningblocks.task.uie.model import UIE
from lightningblocks.task.uie.utils import SpanEvaluator, IEDataset


logger = Logger("UIE")


@torch.no_grad()
def evaluate(model, metric, data_loader, device='gpu', loss_fn=None, show_bar=True):
    """
    Given a datasets, it evals model and computes the metrics.
    Args:
        model(obj:`torch.nn.Module`): A model to classify texts.
        metric(obj:`Metric`): The evaluation metrics.
        data_loader(obj:`torch.utils.data.DataLoader`): The datasets loader which generates batches.
    """
    return_loss = loss_fn is not None
    model.eval()
    metric.reset()
    loss_list = []
    loss_sum = 0
    loss_num = 0
    if show_bar:
        data_loader = tqdm(data_loader, desc="Evaluating", unit='batch')
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
        if device == 'gpu':
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            att_mask = att_mask.cuda()
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask)

        start_prob, end_prob = outputs[0], outputs[1]
        if device == 'gpu':
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
        start_ids = start_ids.type(torch.float32)
        end_ids = end_ids.type(torch.float32)
        if return_loss:
            loss_start = loss_fn(start_prob, start_ids)
            loss_end = loss_fn(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss = float(loss)
            loss_list.append(loss)
            loss_sum += loss
            loss_num += 1
            if show_bar:
                data_loader.set_postfix({'dev losses': f'{loss_sum / loss_num:.5f}'})
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)

        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return (sum(loss_list) / len(loss_list), precision, recall, f1) if return_loss else (precision, recall, f1)


def do_eval(args):
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model = UIE.from_pretrained(args.model_path)
    if args.device == 'gpu':
        model = model.cuda()

    test_ds = IEDataset(args.test_path, tokenizer=tokenizer,
                        max_seq_len=args.max_seq_len)

    test_data_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)
    metric = SpanEvaluator()
    precision, recall, f1 = evaluate(
        model, metric, test_data_loader, args.device)
    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                (precision, recall, f1))


if __name__ == "__main__":
    # yapf: disable
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="The path of saved model that you want to load.")
    parser.add_argument("-t", "--test_path", type=str, required=True,
                        help="The path of test set.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("-D", '--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to run model, defaults to gpu.")

    args = parser.parse_args()

    do_eval(args)
