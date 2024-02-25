import logging
import time

import cv2

import matplotlib.pyplot as plt

from pkg.draw.draw_2d import Draw2d
from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.video_reader.video_reader import VideoReader


def visualize_data(files):
    pose = MediaPipePose()
    draw_keypoints_2d(files, pose)


def draw_keypoints_2d(files, model):
    # cv2.startWindowThread()
    # cv2.namedWindow('preview')
    for file in files:
        logging.info(f"Drawing keypoints for {file}")
        sample_video_reader = VideoReader(file)
        frame_count = sample_video_reader.next_frame()
        draw2D = Draw2d("Keypoints")
        plotImage = Draw2d("Preview")
        while frame_count is not None:
            frame = sample_video_reader.get_current_frame()
            poseLandmarkerResult = model.estimate_frame(frame, int(sample_video_reader.get_frame_timestamp()))
            # key_points = model.extract_keypoints(poseLandmarkerResult)
            angles = model.calculate_keypoint_angle(poseLandmarkerResult.pose_landmarks[0])
            # logging.info(f"key_points: {key_points}")
            logging.info(f"angles: {angles}")
            draw2D.clear_plot()
            draw2D.plot_keypoints(poseLandmarkerResult.pose_landmarks[0])

            annotated_frame = model.draw_landmarks(frame, poseLandmarkerResult)
            plotImage.imshow(annotated_frame)

            # plt.show()
            plt.pause(0.005)
            frame_count = sample_video_reader.next_frame()