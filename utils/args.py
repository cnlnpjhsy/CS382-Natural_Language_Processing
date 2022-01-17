#coding=utf-8
import argparse
from email.policy import default
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--local', default='hfl/chinese-roberta-wwm-ext', help='root of local roberta model')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--output', action='store_true', help='only output results in this mode')
    arg_parser.add_argument('--corrector', action='store_true', help='enable corrector in the model')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--hidden_size', default=768, type=int, help='hidden size')
    arg_parser.add_argument('--slot_loss', default=0.01, type=float, help='weight of slot loss')
    arg_parser.add_argument('--intent_loss', default=1, type=float, help='weight of intent loss')
    return arg_parser