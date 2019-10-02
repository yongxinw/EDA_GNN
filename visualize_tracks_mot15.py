# Created by yongxinwang at 2019-09-16 19:07
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2


sequence = "TUD-Campus"
image_dir = "/hdd/yongxinw/2DMOT2015/train/{}/img1/".format(sequence)
total_frames = len(os.listdir(image_dir))
exp_dir = "/hdd/yongxinw/MOT17/experiments/train_mot15/validation/{}".format(sequence)
if not os.path.exists(os.path.join(exp_dir, "images2")):
    os.makedirs(os.path.join(exp_dir, "images2"))
# track_file = "Sep-18-at-12-57.txt"
# track_file = "Sep-18-at-16-13.txt"
# track_file = "Sep-18-at-17-15.txt"
# track_file = "Sep-18-at-19-51.txt"
# track_file = "Sep-23-at-20-56.txt"
# track_file = "Sep-23-at-20-56.txt"
# tracks = np.loadtxt(os.path.join(exp_dir, track_file))

# tracks = np.loadtxt("/hdd/yongxinw/MOT17/new_experiments/test09_lr0.001/tracks/09/Sep-24-at-12-37.txt")
tracks = np.loadtxt("/hdd/yongxinw/MOT17/experiments/train_mot15/validation/TUD-Campus.txt")

for i, frame in enumerate(range(1, total_frames)):
    # curr_tracks = tracks[(tracks[:, 0] == frame) & (tracks[:, 7] == 1) & (tracks[:, 8] >= 0.6)]
    curr_tracks = tracks[(tracks[:, 0] == frame)]
    # curr_tracks = curr_tracks[curr_tracks[:, 5] > 120]
    # curr_tracks = curr_tracks[:25]
    boxes = curr_tracks[:, 2:6]
    print(len(boxes))
    image = cv2.imread(os.path.join(image_dir, "%06d.jpg" % frame))

    print(image.shape)
    for j, box in enumerate(boxes):
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        id = curr_tracks[j, 1]
        # cls = curr_tracks[j, 7]
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 255))
        cv2.putText(image, str(id), (int(x0), int(y0-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

        # cv2.putText(image, "cls: "+str(cls), (int(x0 + 70), int(y0 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    cv2.imwrite(os.path.join(exp_dir, "images2", "track-%06d.jpg" % frame), image)

os.system("ffmpeg -framerate 15 -i {}/{}/track-%06d.jpg -c:v libx264 "
          "-profile:v high -crf 20 -pix_fmt yuv420p {}/{}/{}-det.avi".format(exp_dir, "images2", exp_dir, "images2", sequence))
