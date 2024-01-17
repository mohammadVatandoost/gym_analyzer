import logging
import os
import string

import numpy as np
import keras

from pkg.dataset.utils import directory_list, mp4_files_list


class ExerciseVideoData:
    def __init__(self, exercise_type, files_name) -> None:
        self.exercise_type = exercise_type
        self.files_name = files_name



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
    logging.info(f"labels = {label_processor.get_vocabulary()}")
    return videos, label_processor
