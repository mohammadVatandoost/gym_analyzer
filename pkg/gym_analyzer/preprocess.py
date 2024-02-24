
import cv2
import numpy as np
import keras

from pkg.dataset.dataset import ExerciseVideoData
from pkg.gym_analyzer.keypoint import FeatureExtractor
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def prepare_all_videos(feature_extractor, exercise_videos: list[ExerciseVideoData], label_processor):
    num_samples = len(exercise_videos)
    # video_paths = df["video_name"].values.tolist()
    # labels = df["tag"].values
    labels = []

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, exercise_video in enumerate(exercise_videos):
        # Gather all its frames and add a batch dimension.
        labels.append(exercise_video.exercise_type)
        frames = load_video(exercise_video.file_name)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :], verbose=0,
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    labels = label_processor(labels)
    # labels = keras.ops.convert_to_numpy(labels)
    return (frame_features, frame_masks), labels.numpy()


def preprocess_on_key_points(feature_extractor, exercise_videos: list[ExerciseVideoData], label_processor, data_path, sequence_length):
    key_point_extractor = FeatureExtractor(exercise_videos,  feature_extractor,  sequence_length, label_processor, data_path)
    sequences, labels, key_point_path = key_point_extractor.extract()
    X = np.array(sequences)
    # Y = np.array(labels)
    Y = to_categorical(labels).astype(int)
    return X, Y

