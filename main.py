# -*- coding: utf-8 -*-
# @File    : main.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/9/27

import os.path as osp
from utils import *
# from train import train_EmbeddingNet
# from train import train_LSTM
# from train import train_FuckUpNet
from train import train_net_1024, train_net_1024_detection
import sys
from Generator import Generator
from Generator_MOT15 import GeneratorMOT15
from Generator_MOT15_Anchors import GeneratorMOT15Anchor
import numpy as np
import random

seed = 1
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

config = osp.join(os.path.abspath(os.curdir), "config.yml")
parser, settings_show = Config(config)
os.environ["CUDA_VISIBLE_DEVICES"] = parser.device

if parser.mode == 0:
    log_path = osp.join(parser.result, 'train_mot15_w_detect_3anchors')
else:
    log_path = osp.join(parser.result, '{}-{}'.format(time_for_file(), parser.description))

os.makedirs(log_path, exist_ok=True)

log = open(osp.join(log_path, 'log.log'), 'w')

print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
print_log("torch version : {}".format(torch.__version__), log)
print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
for idx, data in enumerate(settings_show):
    print_log(data, log)

# generator = Generator(entirety=parser.entirety)
# generator = GeneratorMOT15(entirety=parser.entirety)
generator = GeneratorMOT15Anchor(parser=parser, entirety=parser.entirety, val=False)


if parser.model == "EmbeddingNet":
    train_EmbeddingNet.train(parser, generator, log, log_path)
elif parser.model == "lstm":
    train_LSTM.train(parser, generator, log, log_path)
elif parser.model == "FuckUpNet":
    train_FuckUpNet.train(parser, generator, log, log_path)
elif parser.model == "net_1024":
    train_net_1024.train(parser, generator, log, log_path)
elif parser.model == "net_1024_detect":
    train_net_1024_detection.train(parser, generator, log, log_path)
else:
    raise NotImplementedError

log.close()
