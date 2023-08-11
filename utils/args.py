import argparse

def parse():
    parser = argparse.ArgumentParser(description="一个简单的命令行参数示例")
    
    # 添加一个命令行参数
    parser.add_argument('--stage', type=str, required=True, help="stage of the intermedia result, first or second or third or ")
    parser.add_argument('--num', type=int, required=False, help="number of batches, the test images, greater than 0, less than 10000")
    parser.add_argument('--batch_size', type=int, required=False, help="just batch_size, default is 100")
    parser.add_argument('--display_freq', type=int, required=False, help="display frequency, default is 1")
    parser.add_argument('--time', type=bool, required=False, help="whether to measure the time of the intermedia result, default is False")
    # 解析命令行参数
    args = parser.parse_args()
    check(args)
    if(args.num is None) :
        print(f"now you test the {args.stage} stage, using all(10000) test images!")
    else:
        print(f"now you test the {args.stage} stage, using {args.num * args.batch_size} test images!")
    # 使用命令行参数
    return args

def check(args):
    if(args.time is None):
        args.time = False
    if(args.display_freq is None):
        args.display_freq = 1
    if(args.batch_size is None):
        args.batch_size = 100
    if(args.stage != "first" and args.stage != "second" and args.stage != "third"):
        print("stage must be first or second or third")
        exit()
    if(args.num is None):
        return
    elif(args.num <= 0):
        print("num must be greater than 0")
        exit()
    elif(args.num > 10000):
        print("num must be less than 10000")
        exit()
    pass