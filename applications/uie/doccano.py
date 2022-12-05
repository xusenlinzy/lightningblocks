from lightningnlp.task.uie import convert_data, parse_doccano_args

if __name__ == '__main__':
    args = parse_doccano_args()
    convert_data(args)
