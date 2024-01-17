import logging
import numpy as np
import torch
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
        #  humans = [BodyPart:0-(0.61, 0.34) score=0.95 BodyPart:1-(0.60, 0.38) score=0.94 BodyPart:2-(0.58, 0.38) score=0.96 BodyPart:3-(0.58, 0.42) score=0.89 BodyPart:4-(0.59, 0.47) score=0.74 BodyPart:5-(0.62, 0.38) score=0.95 BodyPart:6-(0.61, 0.43) score=0.87 BodyPart:7-(0.61, 0.47) score=0.78 BodyPart:8-(0.58, 0.44) score=0.84 BodyPart:9-(0.57, 0.49) score=0.83 BodyPart:10-(0.57, 0.55) score=0.91 BodyPart:11-(0.60, 0.45) score=0.79 BodyPart:12-(0.61, 0.49) score=0.78 BodyPart:13-(0.61, 0.55) score=0.85 BodyPart:14-(0.60, 0.34) score=0.98 BodyPart:15-(0.61, 0.34) score=0.97 BodyPart:16-(0.59, 0.34) score=0.92 BodyPart:17-(0.61, 0.34) score=0.34, BodyPart:0-(0.81, 0.38) score=0.86 BodyPart:1-(0.86, 0.39) score=0.87 BodyPart:2-(0.88, 0.36) score=0.79 BodyPart:5-(0.83, 0.42) score=0.81 BodyPart:6-(0.82, 0.55) score=0.85 BodyPart:7-(0.80, 0.64) score=0.89 BodyPart:8-(0.92, 0.60) score=0.61 BodyPart:9-(0.86, 0.68) score=0.67 BodyPart:10-(0.87, 0.83) score=0.77 BodyPart:11-(0.88, 0.60) score=0.63 BodyPart:12-(0.79, 0.71) score=0.82 BodyPart:13-(0.80, 0.89) score=0.80 BodyPart:15-(0.80, 0.37) score=0.88 BodyPart:17-(0.82, 0.36) score=0.91, BodyPart:0-(0.10, 0.16) score=1.01 BodyPart:1-(0.12, 0.22) score=0.81 BodyPart:2-(0.13, 0.23) score=0.67 BodyPart:5-(0.11, 0.22) score=0.83 BodyPart:6-(0.06, 0.18) score=0.84 BodyPart:7-(0.04, 0.10) score=0.88 BodyPart:8-(0.14, 0.38) score=0.69 BodyPart:9-(0.13, 0.51) score=0.79 BodyPart:10-(0.15, 0.62) score=0.89 BodyPart:11-(0.11, 0.39) score=0.72 BodyPart:12-(0.12, 0.52) score=0.86 BodyPart:13-(0.13, 0.65) score=0.85 BodyPart:15-(0.11, 0.16) score=0.99 BodyPart:17-(0.12, 0.18) score=0.94]
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

