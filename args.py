import argparse


def getArgs():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--batchsize', default=10, type=int, help='the bacth size')
    parser.add_argument('--batch', default=10, type=int, help='the bacth size')
    parser.add_argument('--adv_tgr', default='your_adv_path', type=str, help='the path for adversarial examples')
    parser.add_argument('--result_path', default='results.txt', type=str,
                        help='the results of evaluation')

    parser.add_argument('--label_path', default='./labels_1000.txt', type=str,
                        help='label path')
    return parser.parse_args()