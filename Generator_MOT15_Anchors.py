# -*- coding: utf-8 -*-
# @File    : Generator.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/10/11

import os, random
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import build_anchors, np_vec_iou, visualize_boxes, random_colors, Config

np.set_printoptions(suppress=True)

def LoadImg(img_path):
    path = os.listdir(img_path)
    path.sort()
    imglist = []

    for i in range(len(path)):
        img = Image.open(osp.join(img_path, path[i]))
        imglist.append(img.copy())
        img.close()

    return imglist


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


def FindMatchMultiple(list_id, gtid, anchor_ids):
    """
    Find matching indices between gt and anchors
    :param list_id: id's shared by gt and anchors
    :param gtid: list of gt id's
    :param anchor_ids: list of anchor ids
    :return:
    """
    index_pair = []
    for index, id in enumerate(list_id):
        index_gt = gtid.index(id)
        list_indices_anchors = np.where(np.array(anchor_ids) == id)[0]
        for index_anchor in list_indices_anchors:
            index_pair.extend([index_gt, index_anchor])

    return index_pair


class VideoData(object):

    def __init__(self, seq_id, parser):
        # self.img = LoadImg("2DMOT2015/2DMOT2015/train/2DMOT2015-{}-SDP/img1".format(seq_id))
        # self.gt = np.loadtxt("2DMOT2015/label/{}_gt.txt".format(seq_id))

        self.img = LoadImg("/hdd/yongxinw/2DMOT2015/train/{}/img1".format(seq_id))
        self.gt = np.loadtxt("/hdd/yongxinw/2DMOT2015/train/{}/gt/gt.txt".format(seq_id), delimiter=",")

        self.ImageWidth = self.img[0].size[0]
        self.ImageHeight = self.img[0].size[1]

        self.transforms = transforms.Compose([
            transforms.Resize((84, 32)),
            transforms.ToTensor()
        ])

        # self.anchor_heights = [84, 168, 252]
        # self.anchor_widths = [32, 64, 96]
        self.anchor_heights = parser.anchor_heights
        self.anchor_widths = parser.anchor_widths
        self.start_ct = parser.start_ct
        self.start_cl = parser.start_cl

        self.gapH = parser.gapH
        self.gapW = parser.gapW

        self.iou_pos_threshold = parser.iou_pos_threshold
        self.iou_neg_threshold = parser.iou_neg_threshold

        self.anchors = self.generate_anchors()
        # print(self.anchors)
        print(self.anchors.shape)
        # self.visualize_anchors()
        # for i in range(0, len(self.img)):
        #     self.build_targets(i, debug=True)
        # self.build_targets(0, debug=True)

        self.classes = parser.classes

        # note: added by yongxinw
        self.name = "2DMOT2015-{}".format(seq_id)

    def generate_anchors(self):
        anchors = []
        for aw, ah in zip(self.anchor_widths, self.anchor_heights):
            print(aw, ah, self.ImageWidth, self.ImageHeight)
            anchors.append(
                build_anchors(self.start_ct - ah // 2, self.start_cl - aw // 2, aw, ah,
                              self.ImageWidth, self.ImageHeight, gapH=self.gapH, gapW=self.gapW))
        return np.concatenate(anchors)

    def visualize_anchors(self):
        image = self.img[0]
        image = visualize_boxes(image, self.anchors, width=5, outline="green")
        os.makedirs("/hdd/yongxinw/Det/debug", exist_ok=True)
        image.save(osp.join("/hdd/yongxinw/Det/debug", "anchors.jpg"))

    def get_gt_data(self, frame):
        # only consider the eligible ground truth boxes as per Table 2 in https://arxiv.org/pdf/1504.01942.pdf
        gt = self.gt[(self.gt[:, 0] == (frame + 1)) & (self.gt[:, 6] == 1)]
        return gt

    def build_targets(self, frame, debug=False):
        """
        assign anchors to corresponding ground truth targets
        :param frame: frame index
        :param debug: debug mode
        :return:
        """
        gt_data = self.get_gt_data(frame)
        gt_boxes = gt_data[:, 2:6]

        IoU = np_vec_iou(self.anchors, gt_boxes)
        # Initialize target index as -1's (assign index=-1 for negative anchors)
        target_index = np.zeros(self.anchors.shape[0], dtype=np.int) - 1
        # Assign positive anchor box to the gt target with whom the anchor box has the maxIoU >= self.iou_pos_threshold
        pos_mask = np.max(IoU, axis=1) >= self.iou_pos_threshold  # mask of the positive anchors
        target_index[pos_mask] = np.argmax(IoU, axis=1)[pos_mask]

        # todo: deal with thresholded negative indices
        neg_mask = np.max(IoU, axis=1) < self.iou_neg_threshold

        # Assign the gt target's class label to the anchor
        anchor_class = target_index.copy()
        # background_mask = anchor_class == -1
        anchor_class[~pos_mask] = 0  # Backgroud
        anchor_class[pos_mask] = 1  # Pedestrian

        # Calculate the bbox offsets from POSITIVE anchors to their assigned target gt
        # according to SSD equation (2). https://arxiv.org/pdf/1512.02325.pdf
        targets = gt_boxes[target_index]
        ll_norm = (targets[:, 0] - self.anchors[:, 0]) / self.anchors[:, 2]  # g_hat_x = (g_x - d_x) / d_w
        rr_norm = (targets[:, 1] - self.anchors[:, 1]) / self.anchors[:, 3]  # g_hat_y = (g_y - d_y) / d_h
        w_norm = np.log(targets[:, 2] / self.anchors[:, 2])  # g_hat_w = log(g_w / d_w)
        h_norm = np.log(targets[:, 3] / self.anchors[:, 3])  # g_hat_h = log(g_h / d_h)
        # concatenate and mask only the positive anchors
        offsets = np.concatenate((
            ll_norm.reshape(-1, 1),
            rr_norm.reshape(-1, 1),
            w_norm.reshape(-1, 1),
            h_norm.reshape(-1, 1)), axis=1)
        offsets *= pos_mask.astype(np.int).reshape(-1, 1)

        if debug:
            image = self.img[frame]
            gt_ids = gt_data[:, 1]
            max_color = 5
            colors = random_colors(max_color, bright=True)
            image = visualize_boxes(image, self.anchors, width=1, outline="white")

            # Visualize gt boxes
            for i, gt_box in enumerate(gt_boxes):
                id = gt_ids[i]
                color_tmp = tuple([int(tmp * 255) for tmp in colors[int(id % max_color)]])
                image = visualize_boxes(image, [gt_box], width=5, outline=color_tmp)

            colors = random_colors(max_color, bright=False)

            # Visualize anchors and targets
            for j, anchor_box in enumerate(self.anchors):
                target_j = int(target_index[j])
                if target_j != -1:
                    id = gt_ids[target_j]
                    color_tmp = tuple([int(tmp * 255) for tmp in colors[int(id % max_color)]])
                    width = 2
                    # offset_j = offsets[j]
                    # gw, gh = np.exp(offset_j[2:]) * anchor_box[2:]
                    # gleft, gtop = offset_j[:2] * anchor_box[2:] + anchor_box[:2]
                    # anchor_box_aligned = [gleft, gtop, gw, gh]
                    # image = visualize_boxes(image, [anchor_box_aligned], width=width, outline=color_tmp)
                    image = visualize_boxes(image, [anchor_box], width=width, outline=color_tmp)
            # save images for debug
            os.makedirs("/hdd/yongxinw/Det/debug", exist_ok=True)
            image.save(osp.join("/hdd/yongxinw/Det/debug", "anchors_{:03d}.jpg".format(frame + 1)))

        return pos_mask, neg_mask, gt_data, target_index, anchor_class, offsets, self.anchors

    def CurData(self, frame, debug=False):
        pos_mask, neg_mask, gt_data, target_index, anchor_class, offsets, anchors = self.build_targets(frame, debug)

        return pos_mask, neg_mask, gt_data, target_index, anchor_class, offsets, anchors

    def PreData(self, frame):
        DataList = []
        for i in range(5):
            # only consider the eligible ground truth boxes as per Table 2 in https://arxiv.org/pdf/1504.01942.pdf
            data = self.gt[(self.gt[:, 0] == (frame + 1 - i)) & (self.gt[:, 6] == 1)]
            DataList.append(data)

        return DataList

    def TotalFrame(self):
        frames = np.sort(np.unique(self.gt[:, 0]))[6:]

        return len(self.img)

    def CenterCoordinate(self, SingleLineData):
        x = (SingleLineData[2] + (SingleLineData[4] / 2)) / float(self.ImageWidth)
        y = (SingleLineData[3] + (SingleLineData[5] / 2)) / float(self.ImageHeight)

        return x, y

    def BoxCenterCoordinate(self, SingleLineData):
        x = (SingleLineData[0] + (SingleLineData[2] / 2)) / float(self.ImageWidth)
        y = (SingleLineData[1] + (SingleLineData[3] / 2)) / float(self.ImageHeight)

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

    def BoxAppearance(self, frame, box):
        """
        get the appearance data given frame and box coords
        :param frame: frame index
        :param box: box coordinates N x (left, top, width, height)
        :return:
        """

        appearace = []
        img = self.img[frame]
        for i in range(box.shape[0]):
            crop = img.crop((int(box[i, 0]), int(box[i, 1]), int(box[i, 0]) + int(box[i, 2]),
                             int(box[i, 1]) + int(box[i, 3])))
            crop = self.transforms(crop)
            # crop = crop.unsqueeze(0)
            appearace.append(crop)

        return appearace

    def CurMotion(self, data):
        motion = []
        for i in range(data.shape[0]):
            coordinate = torch.zeros([2])
            # coordinate[0], coordinate[1] = self.CenterCoordinate(data[i])
            coordinate[0], coordinate[1] = self.BoxCenterCoordinate(data[i])
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

    def get_anchor_id(self, target_index, gt_data):
        anchor_ids = []
        for i, ind in enumerate(target_index):
            if ind != -1:
                anchor_gtid = gt_data[ind][1]  # assign the target's pedestrian id to this anchor
            else:
                anchor_gtid = -1  # -1 for negative anchor

            anchor_ids.append(anchor_gtid)
        return anchor_ids

    def __call__(self, frame, debug=False, validation=False):
        """

        :param frame:
        :return:
        """
        assert frame >= 5 and frame < self.TotalFrame()
        # cur = self.CurData(frame)
        pos_mask, neg_mask, gt_data, target_index, anchor_class, offsets, anchors = self.CurData(frame, debug=debug)
        pre = self.PreData(frame - 1)

        # cur_crop = self.Appearance(cur)
        # pre_crop = self.Appearance(pre[0])

        # extract appearance features for the anchors and previous 1 frame
        # visual input to regression/classification head for offsets/anchor_class, and gt_matrix
        cur_crop = self.BoxAppearance(frame, anchors)
        pre_crop = self.BoxAppearance(frame-1, pre[0][:, 2:6])

        # visualize crops for debug
        # if debug:
        #     from torchvision.utils import save_image
        #
        #     save_image(torch.cat(cur_crop, dim=3), filename=osp.join("/hdd/yongxinw/Det/debug", "crops_{:03d}.jpg".format(frame + 1)), nrow=1, padding=2)
        #
        #     im = Image.open(osp.join("/hdd/yongxinw/Det/debug", "crops_{:03d}.jpg".format(frame + 1)))
        #     draw = ImageDraw.Draw(im)
        #     print(im.size)
        #     max_color = 5
        #     colors = random_colors(max_color, bright=True)
        #
        #     for i, idx in enumerate(target_index):
        #         if idx != -1:
        #             gt_id = gt_data[idx][1]
        #             color_tmp = tuple([int(tmp * 255) for tmp in colors[int(gt_id % max_color)]])
        #             box = [i*32, 0, i*32 + 32, 84]
        #             draw.rectangle(box, width=3, outline=color_tmp)
        #     im.save(osp.join("/hdd/yongxinw/Det/debug", "verify_{:03d}.jpg".format(frame + 1)))

        # extract motion features for anchors and previous frames (using gt)
        # motion input to regression/classification head for offsets/anchor_class, and gt_matrix
        cur_motion = self.CurMotion(anchors)
        pre_motion = self.PreMotion(pre)

        # construct the ground truth adjacency matrix
        # get the gt id's assigned to each anchor box, assign -1 for negative box
        cur_id = self.get_anchor_id(target_index, gt_data)
        # cur_id = self.GetID(cur)
        pre_id = self.GetID(pre[0])

        list_id = [x for x in pre_id if x in cur_id]
        index_pair = FindMatchMultiple(list_id, pre_id, cur_id)
        gt_matrix = np.zeros([len(pre_id), len(cur_id)])
        for i in range(len(index_pair) // 2):
            gt_matrix[index_pair[2 * i], index_pair[2 * i + 1]] = 1

        if not validation:
            return cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets
        else:
            curr_frame = self.img[frame]
            prev_frame = self.img[frame-1]
            prev_data = pre[0]
            curr_data = gt_data
            return cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets, \
                   curr_frame, curr_data, prev_frame, prev_data, anchors


class GeneratorMOT15Anchor(object):
    def __init__(self, parser, entirety=False, val=False):
        """

        :param entirety:
        """
        self.sequence = []

        self.SequenceID = parser.SequenceID

        self.validation = val

        self.vis_save_path = "2DMOT2015/visualize"

        print("\n-------------------------- initialization --------------------------")
        for id in self.SequenceID:
            print("initializing sequence {} ...".format(id))
            self.sequence.append(VideoData(id, parser=parser))
            print("initialize {} done".format(id))
        print("------------------------------ done --------------------------------\n")

    def visualize(self, seq_ID, frame, save_path=None):
        """

        :param seq_ID:
        :param frame:
        :param save_path:
        """
        if save_path is None:
            save_path = self.vis_save_path

        print("visualize sequence {}: frame {}".format(self.SequenceID[seq_ID], frame + 1))
        print("video solution: {} {}".format(self.sequence[seq_ID].ImageWidth, self.sequence[seq_ID].ImageHeight))
        cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix = self.sequence[seq_ID](frame)

        for i in range(len(cur_crop)):
            img = cur_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            # draw.text((0, 0), "id: {}\ncoord: {:3.2f}, {:3.2f}".format(int(cur_id[i]), cur_motion[i][0].item(),
            #                                                            cur_motion[i][1].item()), fill=(255, 0, 0))
            img.save(osp.join(save_path, "cur_crop_{}.png".format(str(i).zfill(2))))

        for i in range(len(pre_crop)):
            img = pre_crop[i]
            img = transforms.functional.to_pil_image(img)
            img = transforms.functional.resize(img, (420, 160))
            draw = ImageDraw.Draw(img)
            # draw.text((0, 0), "id: {}\ncoord: {:3.2f}, {:3.2f}".format(int(pre_id[i]), pre_motion[i][4, 0].item(),
            #                                                            pre_motion[i][4, 1].item()), fill=(255, 0, 0))
            img.save(osp.join(save_path, "pre_crop_{}.png".format(str(i).zfill(2))))

        np.savetxt(osp.join(save_path, "gt_matrix.txt"), gt_matrix, fmt="%d")
        np.savetxt(osp.join(save_path, "pre_id.txt"), np.array(pre_id).transpose(), fmt="%d")
        np.savetxt(osp.join(save_path, "cur_id.txt"), np.array(cur_id).transpose(), fmt="%d")

    def __call__(self, debug=False, frame=None):
        """

        :return:
        """
        if not self.validation:
            seq = random.choice(self.sequence)
            if frame is None:
                if seq.name == "2DMOT2015-KITTI-13":
                    valid_frames = list(range(10, 133)) + list(range(212, 340))
                    frame = valid_frames[np.random.choice(len(valid_frames))]
                else:
                    frame = random.randint(5, seq.TotalFrame() - 1)

            # print(seq.name, frame)
            # cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix = seq(frame, debug)
            cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, \
            neg_mask, anchor_class, offsets = seq(frame, debug)

            return cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, \
                   torch.Tensor(anchor_class).long(), \
                   torch.Tensor(offsets).float()
        else:
            assert frame is not None, "frame number must be provided for validation"
            seq = random.choice(self.sequence)
            cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets, \
            curr_frame, curr_data, prev_frame, prev_data, anchors = seq(frame, debug, validation=True)

            return cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets, \
                   curr_frame, curr_data, prev_frame, prev_data, anchors

    def test_num_peds(self):
        seq_to_peds = {}
        for seq in self.sequence:
            total_frame = seq.TotalFrame()
            seq_to_peds[seq.name] = 0
            for f in range(5, total_frame):
                # cur_crop, pre_crop, cur_motion, pre_motion, cur_id, pre_id, gt_matrix = seq(f)
                cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, \
                neg_mask, anchor_class, offsets = seq(f)
                if len(cur_id) > seq_to_peds[seq.name]:
                    seq_to_peds[seq.name] = len(cur_id)
        print(seq_to_peds)


if __name__ == "__main__":
    config = osp.join(os.path.abspath(os.curdir), "config.yml")
    parser, settings_show = Config(config)
    gen = GeneratorMOT15Anchor(parser=parser, entirety=False, val=False)
    gen(frame=6, debug=True)
    # gen.test_num_peds()
