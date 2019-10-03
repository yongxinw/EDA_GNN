# -*- coding: utf-8 -*-
# @File    : train_net_1024.py
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# @Date    : 2018/10/24
import torch.nn.functional as F

import os.path as osp

from model import net_1024
from utils import *
from Generator_MOT15_Anchors import GeneratorMOT15Anchor


def train(parser, generator, log, log_path):
    # print("training net_1024\n")
    # model = net_1024.net_1024()

    print("training final\n")
    model = net_1024.net_1024()

    # "----------------- pretrained model loading -----------------"
    # print("loading pretrained model")
    # checkpoint = torch.load("/home/lallazhao/MOT/result/Oct-25-at-02-17-net_1024/net_1024_88.4.pth")
    # checkpoint = torch.load("/hdd/yongxinw/MOT17/experiments/debug1/net_1024.pth")
    # model.load_state_dict(checkpoint["state_dict"])
    # "------------------------------------------------------------"

    model = model.cuda()
    net_param_dict = model.parameters()

    weight = torch.Tensor([10])
    criterion_BCE = torch.nn.BCEWithLogitsLoss(pos_weight=weight).cuda()
    criterion_CE = torch.nn.CrossEntropyLoss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()
    criterion_SMOOTHL1 = torch.nn.SmoothL1Loss().cuda()

    if parser.optimizer == "SGD":
        optimizer = torch.optim.SGD(net_param_dict, lr=parser.learning_rate,
                                    momentum=parser.momentum, weight_decay=parser.decay, nesterov=True)
    elif parser.optimizer == "Adam":
        optimizer = torch.optim.Adam(net_param_dict, lr=parser.learning_rate, weight_decay=parser.decay)
    elif parser.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(net_param_dict, lr=parser.learning_rate, weight_decay=parser.decay,
                                        momentum=parser.momentum)
    else:
        raise NotImplementedError

    # Main Training and Evaluation Loop
    start_time, epoch_time = time.time(), AverageMeter()

    Batch_time = AverageMeter()
    Loss = AverageMeter()
    CLoss = AverageMeter()
    RLoss = AverageMeter()
    Acc = AverageMeter()
    Acc_pos = AverageMeter()

    # Initialize visual validation
    val_parser, val_generator, val_log_path = init_visual_validation()

    for epoch in range(parser.start_epoch, parser.epochs):
        all_lrs = adjust_learning_rate(optimizer, epoch, parser.gammas, parser.schedule)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (parser.epochs - epoch))

        # ----------------------------------- train for one epoch -----------------------------------
        batch_time, loss, classification_loss, regression_loss, acc, acc_pos = \
            train_net_1024(model, generator, optimizer, criterion_BCE, criterion_CE, criterion_MSE, criterion_SMOOTHL1)

        Batch_time.update(batch_time)
        Loss.update(loss.item())
        CLoss.update(classification_loss.item())
        RLoss.update(regression_loss.item())
        Acc.update(acc)
        Acc_pos.update(acc_pos)

        if epoch % parser.print_freq == 0 or epoch == parser.epochs - 1:
            print_log('Epoch: [{:03d}/{:03d}]\t'
                      'Time {batch_time.val:5.2f} ({batch_time.avg:5.2f})\t'
                      'Match Loss {loss.val:6.3f} ({loss.avg:6.3f})\t'
                      'Cls Loss {closs.val:6.3f} ({closs.avg:6.3f})\t'
                      'Reg Loss {rloss.val:6.3f} ({rloss.avg:6.3f})\t'
                      "Acc {acc.val:6.3f} ({acc.avg:6.3f})\t"
                      "Acc_pos {acc_pos.val:6.3f} ({acc_pos.avg:6.3f})\t".format(
                epoch, parser.epochs, batch_time=Batch_time, loss=Loss, closs=CLoss, rloss=RLoss,
                acc=Acc, acc_pos=Acc_pos), log)

            visual_log(model, epoch, val_parser, val_generator, val_log_path)

            Batch_time.reset()
            Loss.reset()
            CLoss.reset()
            RLoss.reset()


        if (epoch in parser.schedule):
            print_log("------------------- adjust learning rate -------------------", log)
        # -------------------------------------------------------------------------------------------

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if epoch % 100 == 0:
            save_file_path = osp.join(log_path, "net_1024.pth")
            states = {
                "state_dict": model.state_dict(),
            }
            torch.save(states, save_file_path)
    # if parser.save_model:
    #     save_file_path = osp.join(log_path, "net_1024.pth")
    #     states = {
    #         "state_dict": model.state_dict(),
    #     }
    #     torch.save(states, save_file_path)


def train_net_1024(model, generator, optimizer, criterion_BCE, criterion_CE, criterion_MSE, criterion_SMOOTHL1):
    # switch to train mode
    model.train()

    cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets = generator()
    # print(len(cur_crop), cur_crop[0].shape)
    # print(len(pre_crop), pre_crop[0].shape)
    # print(len(cur_motion), cur_motion[0].shape)
    # print(len(pre_motion), pre_motion[0].shape)
    # print(gt_matrix.shape, type(gt_matrix))
    # exit()
    assert len(cur_crop) == len(cur_motion)
    assert len(pre_crop) == len(pre_motion)

    target = torch.from_numpy(gt_matrix).cuda().float().view(-1)

    end = time.time()

    s0, s1, s2, s3, adj1, adj, box_pred, cls_pred = model(pre_crop, cur_crop, pre_motion, cur_motion)
    loss = criterion_BCE(s0, target)
    loss += criterion_BCE(s1, target)
    loss += criterion_BCE(s2, target)
    loss += criterion_BCE(s3, target)
    # loss += matrix_loss(adj1, gt_matrix, criterion_CE, criterion_MSE)
    # loss += matrix_loss(adj, gt_matrix, criterion_CE, criterion_MSE)

    # Post process class predictions (i.e. keep pos_mask + neg_mask, and balance pos and negs)
    pos_inds = np.where(pos_mask)[0]
    neg_inds = np.where(neg_mask)[0]
    # randomly sample twice as many negative indices as positive indices
    rand_negs_subsample_inds = np.random.choice(neg_inds.shape[0], size=pos_inds.shape[0]*2, replace=False)
    keep_inds = list(pos_inds) + list(neg_inds[rand_negs_subsample_inds])
    # print(anchor_class[keep_inds])
    # exit()

    classification_loss = criterion_CE(cls_pred[keep_inds], anchor_class[keep_inds].cuda()) * 10
    regression_loss = criterion_SMOOTHL1(box_pred[np.where(pos_mask)], offsets[np.where(pos_mask)].cuda()) * 100

    # add classification and regression loss
    loss += classification_loss
    loss += regression_loss
    # s0, s3, adj = model(pre_crop, cur_crop)
    # loss = criterion_BCE(s0, target)
    # loss = criterion_BCE(s3, target)
    # loss += matrix_loss(adj1, gt_matrix, criterion_CE, criterion_MSE)
    # loss += matrix_loss(adj, gt_matrix, criterion_CE, criterion_MSE)

    # acc, acc_pos = accuracy(s3.clone(), target.clone())
    acc, acc_pos = accuracy2(adj.clone(), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_time = time.time() - end

    return batch_time, loss, classification_loss, regression_loss, acc, acc_pos


def init_visual_validation():
    val_config = osp.join(os.path.abspath(os.curdir), "val_config.yml")
    val_parser, _ = Config(val_config)
    val_generator = GeneratorMOT15Anchor(parser=val_parser, entirety=val_parser.entirety, val=True)
    val_log_path = osp.join(val_parser.result, 'train_mot15_w_detect_3anchors')
    os.makedirs(val_log_path, exist_ok=True)
    return val_parser, val_generator, val_log_path


def visual_log(model, epoch, val_parser, val_generator, val_log_path):
    # define validation information

    model.eval()
    image_dir = osp.join(val_log_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    for it in range(1):
        frame = 6
        cur_crop, pre_crop, cur_motion, pre_motion, gt_matrix, pos_mask, neg_mask, anchor_class, offsets, curr_image, \
        curr_data, prev_image, prev_data, anchors = val_generator(frame=frame)

        with torch.no_grad():
            s0, s1, s2, s3, adj1, adj, box_pred, cls_pred = model(pre_crop, cur_crop, pre_motion, cur_motion)

        # predicted matching score
        adj_sig = torch.sigmoid(adj.detach().cpu())

        if val_parser.use_gt_match:
            adj_sig = torch.from_numpy(gt_matrix)

        # use top k adj scores for match
        scores, match_idx = torch.topk(adj_sig.t(), 1, dim=1)

        # mask the indices that are below the threshold
        match_idx[scores < val_parser.threshold] = -1

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

        # Visualize anchor detection+classification
        # print(cls_pred)
        # print(anchor_class)
        # print(pos_mask)

        # Visualize detections
        curr_boxes = curr_data[:, 2:6]

        curr_image_copy = curr_image.copy()

        colors = random_colors(max_color, bright=False)
        # Draw negative anchors
        for j, anchor_box in enumerate(anchors):
            # predicted class
            cls_j = np.argmax(cls_pred.detach().cpu().numpy()[j])

            # if we are in debug mode and want to use some gt information, specify in config
            if val_parser.use_gt_anchor_class:
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
            if val_parser.use_gt_anchor_class:
                cls_j = anchor_class[j]
            if val_parser.use_gt_offsets:
                offset_j = offsets[j]
            if cls_j == 1:
                match = match_idx[j]
                match_gt_id = gt_ids[match]
                outline = tuple([int(tmp * 255) for tmp in colors[int(match_gt_id % max_color)]])
                if val_parser.show_aligned_anchors:
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
        curr_image_copy.save(osp.join(image_dir, "curr_det_{:03d}_ep{:05d}.jpg".format(frame + 1, epoch)))
    model.train()
