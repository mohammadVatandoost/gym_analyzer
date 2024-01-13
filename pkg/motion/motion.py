import sys
import time

import cv2
import numpy as np
import logging
from pkg.video_reader.video_reader import VideoReader

# try:
#     from cv2 import cuda_OpticalFlow
# except ImportError:
#     print("NVIDIA Optical Flow SDK is required")
#     sys.exit(1)

print(cv2.getBuildInformation())


class MotionDetection:
    def __init__(self, video_reader: VideoReader) -> None:
        self.video_reader = video_reader
        self.frame = video_reader.get_current_frame()

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parameters for Lucas Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Random colors for tracking
        self.color = np.random.randint(0, 255, (100, 3))
        self.mask = np.zeros_like(self.frame)
        self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.frame_gray , mask=None, **self.feature_params)

        self.hsv = np.zeros_like(self.frame)
        self.hsv[..., 1] = 255

        # resize frame
        self.resized_frame = cv2.resize(self.frame, (960, 540))

        # upload resized frame to GPU
        self.gpu_frame = cv2.cuda_GpuMat()
        self.gpu_frame.upload(self.resized_frame)

        # convert to gray
        self.previous_frame = cv2.cvtColor(self.resized_frame, cv2.COLOR_BGR2GRAY)

        # upload pre-processed frame to GPU
        self.gpu_previous = cv2.cuda_GpuMat()
        self.gpu_previous.upload(self.previous_frame)

        # create gpu_hsv output for optical flow
        self.gpu_hsv = cv2.cuda_GpuMat(self.gpu_frame.size(), cv2.CV_32FC3)
        self.gpu_hsv_8u = cv2.cuda_GpuMat(self.gpu_frame.size(), cv2.CV_8UC3)

        self.gpu_h = cv2.cuda_GpuMat(self.gpu_frame.size(), cv2.CV_32FC1)
        self.gpu_s = cv2.cuda_GpuMat(self.gpu_frame.size(), cv2.CV_32FC1)
        self.gpu_v = cv2.cuda_GpuMat(self.gpu_frame.size(), cv2.CV_32FC1)

        # set saturation to 1
        self.gpu_s.upload(np.ones_like(self.previous_frame, np.float32))

    def dense_optical_flow_by_gpu(self):
        frame = self.video_reader.get_current_frame()
        # upload frame to GPU
        self.gpu_frame.upload(frame)
        # resize frame
        self.gpu_frame = cv2.cuda.resize(self.gpu_frame, (960, 540))

        # convert to gray
        gpu_current = cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2GRAY)

        # create optical flow instance
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
            5, 0.5, False, 15, 3, 5, 1.2, 0,
        )
        # calculate optical flow
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
            gpu_flow, self.gpu_previous, gpu_current, None,
        )

        gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
        cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

        # convert from cartesian to polar coordinates to get magnitude and angle
        gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
            gpu_flow_x, gpu_flow_y, angleInDegrees=True,
        )

        # set value to normalized magnitude from 0 to 1
        self.gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)

        # get angle of optical flow
        angle = gpu_angle.download()
        angle *= (1 / 360.0) * (180 / 255.0)

        # set hue according to the angle of optical flow
        self.gpu_h.upload(angle)

        # merge h,s,v channels
        cv2.cuda.merge([self.gpu_h, self.gpu_s, self.gpu_v], self.gpu_hsv)

        # multiply each pixel value to 255
        self.gpu_hsv.convertTo(rtype=cv2.CV_8U, alpha=255.0, beta=0.0, dst=self.gpu_hsv_8u)

        # convert hsv to bgr
        gpu_bgr = cv2.cuda.cvtColor(self.gpu_hsv_8u, cv2.COLOR_HSV2BGR)

        # send original frame from GPU back to CPU
        frame = self.gpu_frame.download()

        # send result from GPU back to CPU
        bgr = gpu_bgr.download()

        # update previous_frame value
        self.gpu_previous = gpu_current

        return frame, bgr

    # def optical_flow_dense(self):
    #     next_frame = self.video_reader.get_current_frame()
    #     next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    #     self.gpu_next_frame.upload(next_frame_gray)
    #     now = round(time.time() * 1000)
    #     # flow = cv2.calcOpticalFlowFarneback(
    #     #     self.gpu_frame,
    #     #     self.gpu_next_frame,
    #     #     None,
    #     #     0.8,
    #     #     3,
    #     #     15,
    #     #     1,
    #     #     5,
    #     #     1.2,
    #     #     0
    #     # )
    #     # flow = cv2.cuda_GpuMat()
    #     # cv2.cuda_FarnebackOpticalFlow
    #     flow = self.gpu_flow.calc(
    #         self.gpu_frame, self.gpu_next_frame, None,
    #     )
    #
    #     execution_time = round(time.time() * 1000) - now
    #     logging.info(f"{execution_time=}")
    #
    #     gpu_flow_x = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
    #     gpu_flow_y = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
    #     cv2.cuda.split(flow, [gpu_flow_x, gpu_flow_y])
    #     # convert from cartesian to polar coordinates to get magnitude and angle
    #     gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
    #         gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    #     )
    #
    #     # set value to normalized magnitude from 0 to 1
    #     gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)
    #
    #     # get angle of optical flow
    #     angle = gpu_angle.download()
    #     angle *= (1 / 360.0) * (180 / 255.0)
    #
    #     # set hue according to the angle of optical flow
    #     self.gpu_h.upload(angle)
    #
    #     # merge h,s,v channels
    #     cv2.cuda.merge([self.gpu_h, self.gpu_s, gpu_v], self.gpu_hsv)
    #
    #     # multiply each pixel value to 255
    #     self.gpu_hsv.convertTo(cv2.CV_8U, 255.0, self.gpu_hsv_8u, 0.0)
    #
    #     # convert hsv to bgr
    #     gpu_bgr = cv2.cuda.cvtColor(self.gpu_hsv_8u, cv2.COLOR_HSV2BGR)
    #     # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #     # self.hsv[..., 0] = ang * 180 / np.pi / 2
    #     # self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #     # bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
    #     self.gpu_frame.upload(next_frame_gray)
    #     # send result from GPU back to CPU
    #     bgr = gpu_bgr.download()
    #     return bgr

    # def optical_flow_nvidia(self):
    #     next_frame = self.video_reader.get_current_frame()
    #     next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    #     self.gpu_next_frame.upload(next_frame_gray)
    #
    #     # Calculate Optical Flow
    #     # flow = self.optical_flow.calc(self.gpu_frame, self.gpu_next_frame, None)
    #     self.gpu_frame = self.gpu_next_frame
    #     return flow
    
    def motion_detect(self):
        next_frame = self.video_reader.get_current_frame()
        diff = cv2.absdiff(self.frame, next_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def optical_flow_Lucas_Kanade(self):
        next_frame = self.video_reader.get_current_frame()
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        if self.p0 is None or len(self.p0) == 0:
            self.frame_gray = next_frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.frame_gray, mask=None, **self.feature_params)
            return None, None, "No features found to track"
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.frame_gray, next_frame_gray,
            self.p0,
            None,
            **self.lk_params
        )

        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        self.frame_gray = next_frame_gray.copy()

        return good_old, good_new, None



    def marks_flow_vectors(self, frame, good_new, good_old):
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(frame)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        img = cv2.add(frame, self.mask )
        self.p0 = good_new.reshape(-1, 1, 2)
        return img

    @staticmethod
    def marks_contours(frame, contours):
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 300:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
