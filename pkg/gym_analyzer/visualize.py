import logging
import time

import cv2

import matplotlib.pyplot as plt
import numpy as np

from pkg.draw.draw_2d import Draw2d
from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.pose.skeleton import angle_connection, angle_connection_labels
from pkg.video_reader.video_reader import VideoReader


def visualize_data(files):
    files_angles = calculate_angles(files)
    draw_angles_2d_plot(files_angles)

    # pose = MediaPipePose()
    # draw_keypoints_2d(files, pose)


def calculate_angles(files):
    pose = MediaPipePose()
    files_angles = []
    for file in files:
        logging.info(f"Calculating angles for {file}")
        sample_video_reader = VideoReader(file)
        frame_count = sample_video_reader.next_frame()
        angles = []
        while frame_count is not None:
            frame = sample_video_reader.get_current_frame()
            poseLandmarkerResult = pose.estimate_image(frame)
            if (
                poseLandmarkerResult is None
                or poseLandmarkerResult.pose_landmarks is None
            ):
                frame_count = sample_video_reader.next_frame()
                logging.error(f"No landmark, frame_count:{frame_count}, file: {file}")
                continue
            angles.append(
                pose.calculate_keypoint_angle(
                    poseLandmarkerResult.pose_landmarks.landmark
                )
            )
            frame_count = sample_video_reader.next_frame()
        files_angles.append(angles)
    return files_angles


def draw_angles_2d_plot(files_angles):
    fig, axs = plt.subplots(4, 2)
    for i in range(len(angle_connection)):
        for file_angles in files_angles:
            y = []
            for angle in file_angles:
                y.append(angle[i])
            x = np.linspace(0, len(y), len(y))
            # plt.plot(x, y)
            axs[int(i / 2), i % 2].plot(x, y)
            axs[int(i / 2), i % 2].set_title(
                f"Angle {angle_connection_labels[i]}", fontsize=12
            )
            axs[int(i / 2), i % 2].set_xlabel("Frame", fontsize=10)
            axs[int(i / 2), i % 2].set_ylabel(
                f"Angle {angle_connection_labels[i]}", fontsize="medium"
            )
            # plt.xlabel('Frame')
            # plt.ylabel(f'Angle {angle_connection_labels[i]}')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.draw()
    plt.savefig(f"angle_connections.png")
    plt.show()
    # plt.cla()
    # plt.clf()


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
            poseLandmarkerResult = model.estimate_frame(
                frame, int(sample_video_reader.get_frame_timestamp())
            )
            # key_points = model.extract_keypoints(poseLandmarkerResult)
            angles = model.calculate_keypoint_angle(
                poseLandmarkerResult.pose_landmarks[0]
            )
            # logging.info(f"key_points: {key_points}")
            logging.info(f"angles: {angles}")
            draw2D.clear_plot()
            draw2D.plot_keypoints(poseLandmarkerResult.pose_landmarks[0])

            annotated_frame = model.draw_landmarks(frame, poseLandmarkerResult)
            plotImage.imshow(annotated_frame)

            # plt.show()
            plt.pause(0.005)
            frame_count = sample_video_reader.next_frame()
