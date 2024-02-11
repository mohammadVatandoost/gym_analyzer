import logging

import cv2

import matplotlib.pyplot as plt

from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.video_reader.video_reader import VideoReader


def visualize_data(files):
    pose = MediaPipePose()
    draw_keypoints_2d(files, pose)

def draw_keypoints_2d(files, model):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    for file in files:
        logging.info(f"Drawing keypoints for {file}")
        sample_video_reader = VideoReader(file)
        frame_count = sample_video_reader.next_frame()
        while frame_count is not None:
            frame = sample_video_reader.get_current_frame()
            poseLandmarkerResult = model.estimate_frame(frame, int(sample_video_reader.get_frame_timestamp()))
            key_points = model.extract_keypoints(poseLandmarkerResult)

            logging.info(f"key_points: {key_points}")
            model.plot_keypoints(poseLandmarkerResult.pose_landmarks[0])
            annotated_frame = model.draw_landmarks(frame, poseLandmarkerResult)
            cv2.imshow('preview', annotated_frame)

            frame_count = sample_video_reader.next_frame()