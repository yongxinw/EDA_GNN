# Created by yongxinwang at 2019-09-16 19:07
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2


def draw_boxes(image, boxes):
    for j, box in enumerate(boxes):
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        id = curr_tracks[j, 1]
        cls = curr_tracks[j, 7]
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 255))
        cv2.putText(image, str(id), (int(x0), int(y0-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

        cv2.putText(image, "cls: "+str(cls), (int(x0 + 70), int(y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    return image

# sequence = "MOT17-02-DPM"
# sequence = "MOT17-13-DPM"
sequence = "MOT17-09-DPM"
image_dir = "/hdd/yongxinw/MOT17/MOT17/train/{}/img1/".format(sequence)
exp_dir = "/hdd/yongxinw/MOT17/experiments/debug9/{}".format(sequence)
track_file = "Sep-18-at-12-41.txt"

tracks = np.loadtxt(os.path.join(exp_dir, track_file))

for i, frame in enumerate(range(7, 525)):
    # curr_tracks = tracks[(tracks[:, 0] == frame) & (tracks[:, 7] == 1) & (tracks[:, 8] >= 0.6)]
    curr_tracks = tracks[(tracks[:, 0] == frame)]
    prev_tracks = tracks[(tracks[:, 0] == frame - 1)]
    # curr_tracks = curr_tracks[curr_tracks[:, 5] > 120]
    # curr_tracks = curr_tracks[:25]
    curr_boxes = curr_tracks[:, 2:6]
    prev_boxes = prev_tracks[:, 2:6]

    curr_image = cv2.imread(os.path.join(image_dir, "%06d.jpg" % frame))
    curr_image = draw_boxes(curr_image, curr_boxes)

    prev_image = cv2.imread(os.path.join(image_dir, "%06d.jpg" % (frame-1)))
    prev_image = draw_boxes(prev_image, prev_boxes)

    image = np.hstack((prev_image, curr_image))
    cv2.imwrite(os.path.join(exp_dir, "links", "track-%06d.jpg" % i), image)

    exit()

os.system("ffmpeg -framerate 15 -i {}/{}/track-%06d.jpg -c:v libx264 "
          "-profile:v high -crf 20 -pix_fmt yuv420p {}/{}/{}-120-det.avi".format(exp_dir, "images", exp_dir, "images", sequence))
