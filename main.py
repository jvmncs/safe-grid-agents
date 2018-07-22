from parsing import prepare_parser

if __name__=='__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    print(args)