from lightningnlp.task.uie import convert_model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", default="uie-base", type=str,
                        help="Directory of input paddle model.\n Will auto download model [uie-base/uie-tiny]")
    parser.add_argument("-o", "--output_model", default="uie_base_pytorch", type=str,
                        help="Directory of output pytorch model")
    args = parser.parse_args()

    convert_model(args.input_model, args.output_model)
