# -*- coding: utf-8 -*-
# @File    : Test.py
# @Author  : Peizhao Li
# @Contact : lipeizhao1997@gmail.com 
# @Date    : 2018/10/6

import os
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import net_1024


def LoadImg(img_path):
    path = os.listdir(img_path)
    path.sort()
    imglist = []

    for i in range(len(path)):
        img = Image.open(osp.join(img_path, path[i]))
        imglist.append(img.copy())
        img.close()

    return imglist


def LoadModel(model, path):
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        model.load_state_dict(checkpoint)

    model.cuda().eval()

    return model


class VideoData(object):

    def __init__(self, info, res_path):
        # MOT15
        # self.img = LoadImg("/hdd/yongxinw/2DMOT2015/train/{}/img1".format(info[0]))
        # self.det = np.loadtxt("/hdd/yongxinw/2DMOT2015/train/{}/det/det.txt".format(info[0]), delimiter=",")

        # # MOT17
        self.img = LoadImg("/hdd/yongxinw/MOT17/MOT17/train/MOT17-{}-SDP/img1".format(info[0]))
        self.det = np.loadtxt("/hdd/yongxinw/MOT17/label/train/MOT17-{}-{}/det/det.txt".format(info[0], info[1]), delimiter=",")

        self.res_path = res_path

        self.ImageWidth = self.img[0].size[0]
        self.ImageHeight = self.img[0].size[1]
        self.transforms = transforms.Compose([
            transforms.Resize((84, 32)),
            transforms.ToTensor()
        ])

    def DetData(self, frame):
        data = self.det[self.det[:, 0] == (frame + 1)]

        return data

    def PreData(self, frame):
        res = np.loadtxt(self.res_path)
        # res = self.det
        DataList = []
        frames = np.unique(res[:, 0])
        frames.sort()
        # print(frames)
        for i in range(5):
            # data = res[res[:, 0] == (frame + 1 - i)]
            frame = frames[-1 - i]
            data = res[res[:, 0] == frame]

            DataList.append(data)
        # print(DataList)
        return DataList

    def TotalFrame(self):
        return len(self.img)

    def CenterCoordinate(self, SingleLineData):
        x = (SingleLineData[2] + (SingleLineData[4] / 2)) / float(self.ImageWidth)
        y = (SingleLineData[3] + (SingleLineData[5] / 2)) / float(self.ImageHeight)

        return x, y

    def Appearance(self, data):
        appearance = []

        img = self.img[int(data[0, 0]) - 1]
        for i in range(data.shape[0]):
            crop = img.crop((int(data[i, 2]), int(data[i, 3]), int(data[i, 2]) + int(data[i, 4]),
                             int(data[i, 3]) + int(data[i, 5])))
            crop = self.transforms(crop)
            appearance.append(crop)

        return appearance

    def DetMotion(self, data):
        motion = []
        for i in range(data.shape[0]):
            coordinate = torch.zeros([2])
            coordinate[0], coordinate[1] = self.CenterCoordinate(data[i])
            motion.append(coordinate)

        return motion

    def PreMotion(self, DataTuple):
        motion = []
        nameless = DataTuple[0]
        for i in range(nameless.shape[0]):
            coordinate = torch.zeros([5, 2])
            identity = nameless[i, 1]
            coordinate[4, 0], coordinate[4, 1] = self.CenterCoordinate(nameless[i])
            # print(identity)

            for j in range(1, 5):
                unknown = DataTuple[j]
                if identity in unknown[:, 1]:
                    coordinate[4 - j, 0], coordinate[4 - j, 1] = self.CenterCoordinate(
                        unknown[unknown[:, 1] == identity].squeeze())
                else:
                    coordinate[4 - j, :] = coordinate[5 - j, :]

            motion.append(coordinate)

        return motion

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())

        return id

    def __call__(self, frame):
        assert frame >= 5 and frame < self.TotalFrame()
        det = self.DetData(frame)
        pre = self.PreData(frame - 1)
        det_crop = self.Appearance(det)
        pre_crop = self.Appearance(pre[0])
        det_motion = self.DetMotion(det)
        pre_motion = self.PreMotion(pre)
        pre_id = self.GetID(pre[0])

        return det_crop, det_motion, pre_crop, pre_motion, pre_id


class TestGenerator(object):

    def __init__(self, res_path, info):
        net = net_1024.net_1024()
        # net_path = "SaveModel/net_1024_beta2.pth"
        # net_path = "/hdd/yongxinw/MOT17/experiments/debug11/net_1024.pth"
        # net_path = "/hdd/yongxinw/MOT17/experiments/train_mot15/net_1024.pth"
        net_path = "/hdd/yongxinw/MOT15/new_experiments/train_mot15_train/checkpoints/net_39500.pth"
        print("------->  loading net_1024")
        print("-----------------> resuming from {}".format(net_path))
        self.net = LoadModel(net, net_path)

        self.sequence = []

        # # MOT17
        print("------->  initializing  MOT17-{}-{} ...".format(info[0], info[1]))
        self.sequence.append(VideoData(info, res_path))
        print("------->  initialize  MOT17-{}-{}  done".format(info[0], info[1]))

        # MOT15
        # print("------->  initializing  MOT15-{}...".format(info[0]))
        # self.sequence.append(VideoData(info, res_path))
        # print("------->  initialize  MOT15-{}  done".format(info[0]))
        # self.vis_save_path = "test/visualize"

    def visualize(self, SeqID, frame, save_path=None):
        """

        :param seq_ID:
        :param frame:
        :param save_path:
        """
        if save_path is None:
            save_path = self.vis_save_path

        print("visualize sequence {}: frame {}".format(self.SequenceID[SeqID], frame + 1))
        print("video solution: {} {}".format(self.sequence[SeqID].ImageWidth, self.sequence[SeqID].ImageHeight))
        det_crop, det_motion, pre_crop, pre_motion, pre_id = self.sequence[SeqID](frame)

        for i in range(len(det_crop)):
            img = det_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), "num: {}\ncoord: {:3.2f}, {:3.2f}".format(int(i), det_motion[i][0].item(),
                                                                        det_motion[i][1].item()), fill=(255, 0, 0))
            img.save(osp.join(save_path, "det_crop_{}.png".format(str(i).zfill(2))))

        for i in range(len(pre_crop)):
            img = pre_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), "num: {}\nid: {}\ncoord: {:3.2f}, {:3.2f}".format(int(i), int(pre_id[i]),
                                                                                pre_motion[i][4, 0].item(),
                                                                                pre_motion[i][4, 1].item()),
                      fill=(255, 0, 0))
            img.save(osp.join(save_path, "pre_crop_{}.png".format(str(i).zfill(2))))

        np.savetxt(osp.join(save_path, "pre_id.txt"), np.array(pre_id).transpose(), fmt="%d")

    def __call__(self, SeqID, frame):
        # frame start with 5, exist frame start from 1
        sequence = self.sequence[SeqID]
        det_crop, det_motion, pre_crop, pre_motion, pre_id = sequence(frame)
        print("network inference")
        with torch.no_grad():
            s0, s1, s2, s3, adj1, adj = self.net(pre_crop, det_crop, pre_motion, det_motion)

        return adj


def FindMatch(list_id, list1, list2):
    """

    :param list_id:
    :param list1:
    :param list2:
    :return:
    """
    index_pair = []
    for index, id in enumerate(list_id):
        index1 = list1.index(id)
        index2 = list2.index(id)
        index_pair.append(index1)
        index_pair.append(index2)

    return index_pair


class TestVideoDataGT(object):

    def __init__(self, seq_id):
        # self.img = LoadImg("MOT17/MOT17/train/MOT17-{}-SDP/img1".format(seq_id))
        # self.gt = np.loadtxt("MOT17/label/{}_gt.txt".format(seq_id))

        self.img = LoadImg("/hdd/yongxinw/MOT17/MOT17/train/MOT17-{}-SDP/img1".format(seq_id))
        self.gt = np.loadtxt("/hdd/yongxinw/MOT17/label/{}_gt.txt".format(seq_id), delimiter=",")

        self.ImageWidth = self.img[0].size[0]
        self.ImageHeight = self.img[0].size[1]

        self.transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def CurData(self, frame):
        data = self.gt[self.gt[:, 0] == (frame + 1)]
        return data

    def PreData(self, frame):
        DataList = []
        for i in range(5):
            data = self.gt[self.gt[:, 0] == (frame + 1 - i)]
            DataList.append(data)

        return DataList

    def TotalFrame(self):

        return len(self.img)

    def CenterCoordinate(self, SingleLineData):
        x = (SingleLineData[2] + (SingleLineData[4] / 2)) / float(self.ImageWidth)
        y = (SingleLineData[3] + (SingleLineData[5] / 2)) / float(self.ImageHeight)

        return x, y

    def Appearance(self, data):
        """

        :param data:
        :return:
        """
        appearance = []
        img = self.img[int(data[0, 0]) - 1]
        for i in range(data.shape[0]):
            crop = img.crop((int(data[i, 2]), int(data[i, 3]), int(data[i, 2]) + int(data[i, 4]),
                             int(data[i, 3]) + int(data[i, 5])))
            crop = self.transforms(crop)
            appearance.append(crop)

        return appearance

    def CurMotion(self, data):
        motion = []
        for i in range(data.shape[0]):
            coordinate = torch.zeros([2])
            coordinate[0], coordinate[1] = self.CenterCoordinate(data[i])
            motion.append(coordinate)

        return motion

    def PreMotion(self, DataTuple):
        """

        :param DataTuple:
        :return:
        """
        motion = []
        nameless = DataTuple[0]
        for i in range(nameless.shape[0]):
            coordinate = torch.zeros([5, 2])
            identity = nameless[i, 1]
            coordinate[4, 0], coordinate[4, 1] = self.CenterCoordinate(nameless[i])
            for j in range(1, 5):
                unknown = DataTuple[j]
                if identity in unknown[:, 1]:
                    coordinate[4 - j, 0], coordinate[4 - j, 1] = self.CenterCoordinate(
                        unknown[unknown[:, 1] == identity].squeeze())
                else:
                    coordinate[4 - j, :] = coordinate[5 - j, :]
            motion.append(coordinate)

        return motion

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1])

        return id

    def __call__(self, frame):
        """

        :param frame:
        :return:
        """
        assert frame >= 5 and frame < self.TotalFrame()
        cur = self.CurData(frame)
        pre = self.PreData(frame - 1)

        # # Note: added by yongxinw
        # # Downsize the data so that they can fit memory. Note: I wonder how they train this with bs=100?
        # print(len(cur), len(pre))
        # import ipdb
        # ipdb.set_trace()
        # if len(cur) >= 25:
        #     cur = cur[np.random.choice(len(cur), size=25, replace=False)]
        #
        # if len(pre) >= 25:
        #     pre = pre[np.random.choice(len(pre), size=25, replace=False)]

        cur_crop = self.Appearance(cur)
        pre_crop = self.Appearance(pre[0])

        cur_motion = self.CurMotion(cur)
        pre_motion = self.PreMotion(pre)

        cur_id = self.GetID(cur)
        pre_id = self.GetID(pre[0])

        list_id = [x for x in pre_id if x in cur_id]
        index_pair = FindMatch(list_id, pre_id, cur_id)
        gt_matrix = np.zeros([len(pre_id), len(cur_id)])
        for i in range(len(index_pair) / 2):
            gt_matrix[index_pair[2 * i], index_pair[2 * i + 1]] = 1

        return cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix

# Note: Added by yongxinw. This is using GT adj and dets to test if the matching algorithm is working
class TestGeneratorGT(object):
    """This class directly outputs ground truth adjacency matrixs between ground truth annotated boxes"""
    def __init__(self, info):
        # net = net_1024.net_1024()
        # net_path = "SaveModel/net_1024_beta2.pth"
        # print("------->  loading net_1024")
        # self.net = LoadModel(net, net_path)

        self.sequence = []

        print("------->  initializing  MOT17-{}-{} ...".format(info[0], info[1]))
        self.sequence.append(TestVideoDataGT(info[0]))
        print("------->  initialize  MOT17-{}-{}  done".format(info[0], info[1]))

        self.vis_save_path = "test/visualize"

    def __call__(self, frame):
        """
        :param SeqID:
        :param frame:
        :return:
        """
        seq = self.sequence[0]
        cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix = seq(frame)
        return gt_matrix
