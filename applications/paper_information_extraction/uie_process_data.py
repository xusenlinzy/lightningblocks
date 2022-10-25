import argparse
from lightningnlp.task.uie import convert_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--doccano_file", default="./data/doccano.json",
                        type=str, help="The doccano file exported from doccano platform.")
    parser.add_argument("-s", "--save_dir", default="./data",
                        type=str, help="The path of data that you wanna save.")
    parser.add_argument("--negative_ratio", default=5, type=int,
                        help="Used only for the extraction task, the ratio of positive and negative samples, "
                             "number of negative samples = negative_ratio * number of positive samples")
    parser.add_argument("--splits", default=[0.9, 0.1, 0.0], type=float, nargs="*",
                        help="The ratio of samples in datasets. [0.8, 0.1, 0.1] means 80%% samples used for training, "
                             "10%% for evaluation and 10%% for test.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str,
                        help="Select task type, ext for the extraction task and cls for the classification task, "
                             "defaults to ext.")
    parser.add_argument("--options", default=["正向", "负向"], type=str, nargs="+",
                        help="Used only for the classification task, the options for classification")
    parser.add_argument("--prompt_prefix", default="情感倾向", type=str,
                        help="Used only for the classification task, the prompt prefix for classification")
    parser.add_argument("--is_shuffle", default=True, type=bool,
                        help="Whether to shuffle the labeled datasets, defaults to True.")
    parser.add_argument("--seed", type=int, default=1000,
                        help="random seed for initialization")

    args = parser.parse_args()

    convert_data(args)
