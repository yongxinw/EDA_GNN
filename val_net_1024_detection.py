# Created by yongxinwang at 2019-10-02 18:21
import torch.nn.functional as F

import os.path as osp
import sys

from model import net_1024
from utils import *
from Generator_MOT15_Anchors import GeneratorMOT15Anchor


def val(parser, generator, log, log_path):
    print("validation \n")
    model = net_1024.net_1024()

    # "----------------- pretrained model loading -----------------"
    # print("loading pretrained model")
    # checkpoint = torch.load("/home/lallazhao/MOT/result/Oct-25-at-02-17-net_1024/net_1024_88.4.pth")
    checkpoint = torch.load("/hdd/yongxinw/Det/experiments/train_mot15_w_detect_3anchors/net_1024.pth")
    model.load_state_dict(checkpoint["state_dict"])
    # "------------------------------------------------------------"

    image_dir = osp.join(log_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    model = model.cuda()
    model.eval()

    for it in range(1):
        frame = 6
        cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets, curr_image, \
        curr_data, prev_image, prev_data, anchors = generator(frame=frame)

        with torch.no_grad():
            s0, s1, s2, s3, adj1, adj, box_pred, cls_pred = model(pre_crop, cur_crop, pre_motion, cur_motion)

        # predicted matching score
        adj_sig = F.sigmoid(adj)

        if parser.use_gt_match:
            adj_sig = torch.from_numpy(gt_matrix).cuda()

        # use top k adj scores for match
        scores, match_idx = torch.topk(adj_sig.t(), 1, dim=1)

        # print(scores)
        # print(match_idx)

        # mask the indices that are below the threshold
        match_idx[scores < parser.threshold] = -1

        # x_inds = torch.arange(match_idx.shape[0]).view(-1, 1).repeat(1, match_idx.shape[1]).view(-1)
        prev_boxes = prev_data[:, 2:6]
        gt_ids = prev_data[:, 1]
        max_color = 10
        colors = random_colors(max_color, bright=True)

        # Visualize gt boxes
        curr_image_copy = prev_image.copy()
        for i, gt_box in enumerate(prev_boxes):
            id = gt_ids[i]
            color_tmp = tuple([int(tmp * 255) for tmp in colors[int(id % max_color)]])
            curr_image_copy = visualize_boxes(curr_image_copy, [gt_box], width=5, outline=color_tmp)

        curr_image_copy.save(osp.join(image_dir, "prev_{:03d}.jpg".format(frame + 1)))

        # Visualize anchor matching
        # curr_image_copy = curr_image.copy()
        # for j, anchor_box in enumerate(anchors):
        #     matches = match_idx[j]
        #     for k in range(len(matches)):
        #         if matches[k] != -1:
        #             match_gt_id = gt_ids[matches[k]]
        #             width = 3
        #             outline = tuple([int(tmp * 255) for tmp in colors[int(match_gt_id % max_color)]])
        #         else:
        #             width = 1
        #             outline = "white"
        #
        #         curr_image_copy = visualize_boxes(curr_image_copy, [anchor_box], width=width, outline=outline)
        # curr_image_copy.save(osp.join(image_dir, "curr_match_{:03d}.jpg".format(frame)))

        # Visualize anchor detection+classification
        print(cls_pred)
        print(anchor_class)
        print(pos_mask)

        # Visualize detections
        curr_boxes = curr_data[:, 2:6]

        curr_image_copy = curr_image.copy()

        colors = random_colors(max_color, bright=False)
        # Draw negative anchors
        for j, anchor_box in enumerate(anchors):
            # predicted class
            cls_j = np.argmax(cls_pred.detach().cpu().numpy()[j])

            # if we are in debug mode and want to use some gt information, specify in config
            if parser.use_gt_anchor_class:
                cls_j = anchor_class[j]

            if cls_j == 0:
                curr_image_copy = visualize_boxes(curr_image_copy, [anchor_box], width=1, outline='white')

        # Draw positive anchors
        for j, anchor_box in enumerate(anchors):
            # predicted class
            cls_j = np.argmax(cls_pred.detach().cpu().numpy()[j])

            # predicted offset
            offset_j = box_pred.detach().cpu().numpy()[j]

            # if we are in debug mode and want to use some gt information, specify in config
            if parser.use_gt_anchor_class:
                cls_j = anchor_class[j]
            if parser.use_gt_offsets:
                offset_j = offsets[j]
            if cls_j == 1:
                match = match_idx[j]
                match_gt_id = gt_ids[match]
                outline = tuple([int(tmp * 255) for tmp in colors[int(match_gt_id % max_color)]])
                if parser.show_aligned_anchors:
                    gw, gh = np.exp(offset_j[2:]) * anchor_box[2:]
                    gleft, gtop = offset_j[:2] * anchor_box[2:] + anchor_box[:2]
                    anchor_box_aligned = [gleft, gtop, gw, gh]
                    curr_image_copy = visualize_boxes(curr_image_copy, [anchor_box_aligned], width=3, outline=outline)
                else:
                    curr_image_copy = visualize_boxes(curr_image_copy, [anchor_box], width=3, outline=outline)

        # visualize the GT
        for i, gt_box in enumerate(curr_boxes):
            id = gt_ids[i]
            color_tmp = tuple([int(tmp * 255) for tmp in colors[int(id % max_color)]])
            curr_image_copy = visualize_boxes(curr_image_copy, [gt_box], width=5, outline=color_tmp)
        curr_image_copy.save(osp.join(image_dir, "curr_det_{:03d}.jpg".format(frame + 1)))


if __name__ == "__main__":
    config = osp.join(os.path.abspath(os.curdir), "val_config.yml")
    parser, settings_show = Config(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.device

    log_path = osp.join(parser.result, 'train_mot15_w_detect_3anchors')

    os.makedirs(log_path, exist_ok=True)

    log = open(osp.join(log_path, 'val_log.log'), 'w')

    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    for idx, data in enumerate(settings_show):
        print_log(data, log)

    generator = GeneratorMOT15Anchor(parser=parser, entirety=parser.entirety, val=True)

    val(parser, generator, log, log_path)

    log.close()
