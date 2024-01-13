import os
import re
import sys
# sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config
from pkg.video_reader.video_reader import VideoReader


# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', help='experiment configure file name',
#                     default='./experiments/vgg19_368x368_sgd.yaml', type=str)
# parser.add_argument('--weight', type=str,
#                     default='pose_model.pth')
# parser.add_argument('opts',
#                     help="Modify config options using the command-line",
#                     default=None,
#                     nargs=argparse.REMAINDER)
# args = parser.parse_args()
#
# # update config file
# update_config(cfg, args)

class OpenPose():
    def __init__(self, video_reader: VideoReader) -> None:
        self.video_reader = video_reader
        wight_path = "model/pose_model.pth"
        self.model = get_model('vgg19')
        self.model.load_state_dict(torch.load(wight_path))
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.float()
        self.model.eval()

    def estimate(self):
        oriImg = self.video_reader.read_frame()

        # shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, self.model, 'rtpose')

        print(im_scale)
        humans = paf_to_pose_cpp(heatmap, paf, cfg)

        out = draw_humans(oriImg, humans)
        return out
        # cv2.imwrite('result.png', out)

