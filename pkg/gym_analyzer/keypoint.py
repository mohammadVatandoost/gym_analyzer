import logging
import os

import numpy as np

from pkg.dataset.dataset import ExerciseVideoData
from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.video_reader.video_reader import VideoReader


class KeypointExtractor:
    def __init__(self, exercise_videos: list[ExerciseVideoData], model, sequence_length, label_processor, data_path):
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
                logging.info(f"Creating {os.path.join(self.data_path, exercise_type)} directory for storing keypoints")
                os.makedirs(os.path.join(self.data_path, exercise_type))

    @property
    def extract(self):
        sequences, labels = [], []
        for idx, exercise_video in enumerate(self.exercise_videos):
            window = []
            if not os.path.exists(os.path.join(self.data_path, exercise_video.exercise_type, str(os.path.basename(exercise_video.file_name)))):
                logging.info(f"Creating {os.path.join(self.data_path, exercise_video.exercise_type, str(os.path.basename(exercise_video.file_name)))} directory for storing keypoints")
                os.makedirs(
                    os.path.join(
                        self.data_path,
                        exercise_video.exercise_type,
                        str(os.path.basename(exercise_video.file_name))
                    )
                )
            else:
                continue
            sample_video_reader = VideoReader(exercise_video.file_name)
            frame_count = sample_video_reader.next_frame()
            while frame_count is not None:
                results = self.model.estimate_frame(sample_video_reader.get_current_frame())
                key_points = self.model.extract_keypoints(results)
                window.append(key_points)
                if len(window) == self.sequence_length:
                    sequences.append(window)
                    labels.append(self.label_processor(exercise_video.exercise_type))
                    npy_path = os.path.join(
                        self.data_path,
                        exercise_video.exercise_type,
                        str(os.path.basename(exercise_video.file_name)),
                        str(frame_count)
                    )
                    np.save(npy_path, window)
                    window = []
                frame_count = sample_video_reader.next_frame()

        return sequences, labels, self.data_path


    def __load_winodws(self, path, label):
