# Created by yongxinwang at 2019-09-16 19:07
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2



sequence = "MOT17-02-DPM"
image_dir = "/hdd/yongxinw/MOT17/MOT17/train/{}/img1/".format(sequence)
exp_dir = "/hdd/yongxinw/MOT17/experiments/debug1/{}".format(sequence)
track_file = "Sep-16-at-19-53.txt"

tracks = np.loadtxt(os.path.join(exp_dir, track_file))

for i, frame in enumerate(range(6, 601)):
    curr_tracks = tracks[tracks[:, 0] == frame]
    boxes = curr_tracks[:, 2:6]

    image = cv2.imread(os.path.join(image_dir, "%06d.jpg" % frame))

    print(image.shape)
    for j, box in enumerate(boxes):
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        id = curr_tracks[j, 1]
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 255))
        cv2.putText(image, str(id), (int(x0), int(y0-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

    cv2.imwrite(os.path.join(exp_dir, "images", "track-%06d.jpg" % i), image)

os.system("ffmpeg -framerate 15 -i {}/{}/track-%06d.jpg -c:v libx264 "
          "-profile:v high -crf 20 -pix_fmt yuv420p {}/{}/{}.mp4".format(exp_dir, "images", exp_dir, "images", sequence))
