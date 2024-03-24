import logging
import os
import time

import cv2
import numpy as np

from pkg.dataset.dataset import ExerciseVideoData
from pkg.dataset.utils import npy_files_list
from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.video_reader.video_reader import VideoReader

show_frames = False
show_error_frames = False


class FeatureExtractor:
    def __init__(
        self,
        exercise_videos: list[ExerciseVideoData],
        model: any,
        sequence_length: int,
        label_processor: any,
        data_path: str,
    ):
        self.exercise_videos = exercise_videos
        self.model = model
        self.sequence_length = sequence_length
        self.label_processor = label_processor
        # Path for exported data, numpy arrays
        self.data_path = data_path
        logging.info(f"Key points Data path: {self.data_path}")
        # make directory if it does not exist yet
        if not os.path.exists(self.data_path):
            logging.info(f"Creating {self.data_path} directory for storing keypoints")
            os.makedirs(self.data_path)
        for _, exercise_type in enumerate(self.label_processor.get_vocabulary()):
            if not os.path.exists(os.path.join(self.data_path, exercise_type)):
                logging.info(
                    f"Creating {os.path.join(self.data_path, exercise_type)} directory for storing keypoints"
                )
                os.makedirs(os.path.join(self.data_path, exercise_type))

    def extract(self):
        sequences, labels = [], []
        not_suitable_videos = []
        total_error_counter = 0
        if show_frames or show_error_frames:
            cv2.startWindowThread()
            cv2.namedWindow("preview")

        for idx, exercise_video in enumerate(self.exercise_videos):
            window = []
            path = os.path.join(
                self.data_path,
                exercise_video.exercise_type,
                str(os.path.basename(exercise_video.file_name)),
            )
            # make directory if it does not exist yet. If it exist, read them from file and go to next
            if not os.path.exists(path):
                logging.info(
                    f"Creating {os.path.join(self.data_path, exercise_video.exercise_type, str(os.path.basename(exercise_video.file_name)))} directory for storing keypoints"
                )
                os.makedirs(
                    os.path.join(
                        self.data_path,
                        exercise_video.exercise_type,
                        str(os.path.basename(exercise_video.file_name)),
                    )
                )
            else:
                logging.info(f"Loading data from {path}")
                sequences_from_storage, labels_from_storage = self.__load_windows(
                    path, self.label_processor(exercise_video.exercise_type)
                )
                sequences.extend(sequences_from_storage)
                labels.extend(labels_from_storage)
                continue
            # extract keypoint from mp4 file and store them in chunks with size of sequence_length
            sample_video_reader = VideoReader(exercise_video.file_name)
            self.model.reset()
            logging.info(f"video fps: {sample_video_reader.get_video_fps()}")
            frame_count = sample_video_reader.next_frame()
            last_frame_timestamp = -1.0
            frame_counter = 0
            error_counter = 0
            while frame_count is not None:
                frame = sample_video_reader.get_current_frame()
                frame_counter = frame_counter + 1
                frame_timestamp = sample_video_reader.get_frame_timestamp()
                if last_frame_timestamp >= frame_timestamp:
                    logging.error(
                        f"timestamp must be monotonically increasing, "
                        f"last_frame_timestamp: {last_frame_timestamp}, "
                        f"frame_timestamp: {frame_timestamp},"
                        f"frame_count: {frame_count}"
                    )
                    frame_count = sample_video_reader.next_frame()
                    error_counter = error_counter + 1
                    continue
                results = self.model.estimate_frame(frame, int(frame_timestamp))
                last_frame_timestamp = frame_timestamp
                if len(results.pose_landmarks) == 0:
                    logging.error(
                        f"no landmark detected, "
                        f"frame_count: {frame_count},frame_timestamp: {frame_timestamp}"
                    )
                    frame_count = sample_video_reader.next_frame()
                    total_error_counter = total_error_counter + 1
                    error_counter = error_counter + 1
                    if show_error_frames:
                        cv2.imshow("preview", frame)
                        cv2.waitKey(1)
                    continue
                # key_points = self.model.extract_keypoints(results)
                if show_frames:
                    annotated_frame = self.model.draw_landmarks(frame, results)
                    cv2.imshow("preview", annotated_frame)
                    # logging.info(f"key_points: {key_points}")
                # window.append(key_points)
                angles = self.model.calculate_keypoint_angle(results.pose_landmarks[0])
                window.append(angles)
                if len(window) == self.sequence_length:
                    sequences.append(window)
                    labels.append(self.label_processor(exercise_video.exercise_type))
                    npy_path = os.path.join(
                        self.data_path,
                        exercise_video.exercise_type,
                        str(os.path.basename(exercise_video.file_name)),
                        str(frame_count),
                    )
                    np.save(npy_path, window)
                    window = []
                frame_count = sample_video_reader.next_frame()
            logging.info(
                f"frame_counter: {frame_counter}, error_counter: {error_counter}"
            )
            if error_counter > 0 and frame_counter / error_counter < 7:
                not_suitable_videos.append(exercise_video.file_name)
                # self.__delete_file(exercise_video.file_name)
        if show_frames:
            cv2.destroyAllWindows()
        logging.info(f"total_error_counter: {total_error_counter}")
        logging.info(
            f"len(not_suitable_videos): {len(not_suitable_videos)}, not_suitable_videos: {not_suitable_videos}"
        )
        return sequences, labels, self.data_path

    @staticmethod
    def __load_windows(path, label):
        files = npy_files_list(path)
        sequences = []
        labels = []
        for f in files:
            window = np.load(f)
            sequences.append(window)
            labels.append(label)
        return sequences, labels

    @staticmethod
    def __delete_file(file_path: str):
        try:
            os.remove(file_path)
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            logging.error(f"failed to delete the file: {file_path}, error: {e}")
