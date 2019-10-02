# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/9/27

import yaml, torch, time, os
from easydict import EasyDict as edict
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import ImageDraw
import colorsys

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    parser = edict(yaml.load(listfile1))
    settings_show = listfile2.read().splitlines()
    return parser, settings_show


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    multiple = 1
    for (gamma, step) in zip(gammas, schedule):
        if (epoch == step):
            multiple = gamma
            break
    all_lrs = []
    for param_group in optimizer.param_groups:
        param_group['lr'] = multiple * param_group['lr']
        all_lrs.append(param_group['lr'])
    return set(all_lrs)


def print_log(print_string, log, true_string=None):
    print("{}".format(print_string))
    if true_string is not None:
        print_string = true_string
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def time_for_file():
    ISOTIMEFORMAT = '%h-%d-at-%H-%M'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def extract_label(matrix):
    index = np.argwhere(matrix == 1)
    target = index[:, 1]
    target = torch.from_numpy(target).cuda()

    return target


def matrix_loss(matrix, gt_matrix, criterion_CE, criterion_MSE):
    index_row_match = np.where([np.sum(gt_matrix, axis=1) == 1])[1]
    index_col_match = np.where([np.sum(gt_matrix, axis=0) == 1])[1]
    index_row_miss = np.where([np.sum(gt_matrix, axis=1) == 0])[1]
    index_col_miss = np.where([np.sum(gt_matrix, axis=0) == 0])[1]

    gt_matrix_row_match = np.take(gt_matrix, index_row_match, axis=0)
    gt_matrix_col_match = np.take(gt_matrix.transpose(), index_col_match, axis=0)

    index_row_match = torch.from_numpy(index_row_match).cuda()
    index_col_match = torch.from_numpy(index_col_match).cuda()

    matrix_row_match = torch.index_select(matrix, dim=0, index=index_row_match)
    matrix_col_match = torch.index_select(matrix.t(), dim=0, index=index_col_match)

    label_row_CE = extract_label(gt_matrix_row_match)
    label_col_CE = extract_label(gt_matrix_col_match)

    loss = criterion_CE(matrix_row_match, label_row_CE)
    loss += criterion_CE(matrix_col_match, label_col_CE)

    if index_row_miss.size != 0:
        index_row_miss = torch.from_numpy(index_row_miss).cuda()
        matrix_row_miss = torch.index_select(matrix, dim=0, index=index_row_miss)
        loss += criterion_MSE(torch.sigmoid(matrix_row_miss), torch.zeros_like(matrix_row_miss))

    if index_col_miss.size != 0:
        index_col_miss = torch.from_numpy(index_col_miss).cuda()
        matrix_col_miss = torch.index_select(matrix.t(), dim=0, index=index_col_miss)
        loss += criterion_MSE(torch.sigmoid(matrix_col_miss), torch.zeros_like(matrix_col_miss))

    return loss


def accuracy(input, target):
    assert input.size() == target.size()

    # print(list(input.detach().cpu().numpy()))

    input[input < 0] = 0
    input[input > 0] = 1
    # print(list(input.detach()log.cpu().numpy()))
    batch_size = input.size(0)
    pos_size = torch.sum(target)

    dis = input.sub(target)
    wrong = torch.sum(torch.abs(dis))
    acc = (batch_size - wrong.item()) / batch_size

    index = torch.nonzero(target)
    input_pos = torch.sum(input[index])
    acc_pos = input_pos.item() / pos_size

    return acc, acc_pos


def accuracy2(input, target):
    assert input.view(-1).size() == target.size()

    # use Hungarian to calculate accuracy
    row_idx, col_idx = linear_sum_assignment(-input.detach())
    input = torch.zeros_like(input)

    input[row_idx, col_idx] = 1
    input = input.view(-1)

    batch_size = input.size(0)
    pos_size = torch.sum(target)

    dis = input.sub(target)
    wrong = torch.sum(torch.abs(dis))
    acc = (batch_size - wrong.item()) / batch_size

    index = torch.nonzero(target)
    input_pos = torch.sum(input[index])
    acc_pos = input_pos.item() / pos_size

    return acc, acc_pos

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# utils for generating anchors
def build_anchors(start_t, start_l, aW, aH, width, height, gapH=20, gapW=20):
    """
    Generate anchors across the image
    :param start_t: starting center pixel location (top)
    :param start_l: starting center pixel location (left)
    :param aW: anchor width
    :param aH: anchor height
    :param width: image width
    :param height: image height
    :return: numpy array of anchors Nx4 (left, top, width, height)
    """

    tops = np.arange((height - start_t) // gapH) * gapH + start_t
    lefts = np.arange((width - start_l) // gapW) * gapW + start_l

    ll, tt = np.meshgrid(lefts, tops)
    tls = np.concatenate((ll.reshape(1, -1), tt.reshape(1, -1))).T
    wh = np.tile([aW, aH], [len(tls), 1])
    tlwh = np.concatenate((tls, wh), axis=1)

    return tlwh


# util for calculating IoU vectorized
def np_vec_iou(bboxes1, bboxes2):
    """
    Vectorized calculation for IoU between 2 sets of boxes
    :param bboxes1: N x (left, top, width, height)
    :param bboxes2: M x (left, top, width, height)
    :return: array of iou NxM
    """
    x1, y1, w1, h1 = np.split(bboxes1, 4, axis=1)
    x2, y2, w2, h2 = np.split(bboxes2, 4, axis=1)

    x11, y11, x12, y12 = x1, y1, x1 + w1, y1 + h1
    x21, y21, x22, y22 = x2, y2, x2 + w2, y2 + h2

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou



def visualize_boxes(image, boxes, **kwargs):
    """
    Visualize boxes on the image
    :param image: PIL image object
    :param boxes: array of boxes N x (left, top, width, height)
    :param kwargs: PIL Image.Draw.draw keyword args (i.e. width, outline)
    :return:
    """
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], **kwargs)
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    :param N: number of colors
    :param bright: whether to draw bright colors
    :return array of colors
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors
