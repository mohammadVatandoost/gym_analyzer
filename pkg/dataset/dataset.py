import logging
import os
import string
from random import shuffle
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import keras

from pkg.dataset.utils import directory_list, mp4_files_list


class ExerciseVideoData:
    def __init__(self, exercise_type, file_name) -> None:
        self.exercise_type = exercise_type
        self.file_name = file_name


def read_datasets(directory):
    directories = directory_list(directory)
    labels = []
    videos = []
    is_first = True
    for dir in directories:
        if is_first:
            is_first = False
            continue

        files = mp4_files_list(dir)
        # dir_name = os.path.dirname(dir)
        dir_name = os.path.basename(dir)
        labels.append(dir_name)
        for f in files:
            v = ExerciseVideoData(dir_name, f)
            videos.append(v)

    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(labels)
    )
    shuffle(videos)
    logging.info(f"labels = {label_processor.get_vocabulary()}")
    data = np.array(videos)
    x_train ,x_test = train_test_split(data, test_size=0.25)
    logging.info(f"quantity of train data: {len(x_train)}, quantity of test data: {len(x_test)}")
    return x_train, x_test, label_processor

