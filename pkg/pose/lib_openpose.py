import logging
import numpy as np
import torch
import cv2
from pkg.pose import im_transform
from pkg.pose.draw import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from pkg.pose.paf_to_pose import paf_to_pose_cpp
from pkg.pose.config import _C as cfg
from pkg.video_reader.video_reader import VideoReader
from pkg.pose.rtpose_vgg import get_model
from pkg.pose.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)

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

        with torch.no_grad():
            paf, heatmap, im_scale = self.get_outputs(oriImg, self.model, 'rtpose')

        # print(im_scale)
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        logging.info(f"humans = {humans}")
        out = draw_humans(oriImg, humans)
        return out
        # cv2.imwrite('result.png', out)

    def get_outputs(self, img, model, preprocess):
        """Computes the averaged heatmap and paf for the given image
        :param multiplier:
        :param origImg: numpy array, the image being processed
        :param model: pytorch model
        :returns: numpy arrays, the averaged paf and heatmap
        """
        inp_size = cfg.DATASET.IMAGE_SIZE

        # padding
        im_croped, im_scale, real_shape = im_transform.crop_with_factor(
            img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

        if preprocess == 'rtpose':
            im_data = rtpose_preprocess(im_croped)

        elif preprocess == 'vgg':
            im_data = vgg_preprocess(im_croped)

        elif preprocess == 'inception':
            im_data = inception_preprocess(im_croped)

        elif preprocess == 'ssd':
            im_data = ssd_preprocess(im_croped)

        batch_images = np.expand_dims(im_data, 0)

        # several scales as a batch
        batch_var = torch.from_numpy(batch_images).cuda().float()
        predicted_outputs, _ = model(batch_var)
        output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
        heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
        paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

        return paf, heatmap, im_scale

