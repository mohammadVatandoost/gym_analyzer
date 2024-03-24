import logging
import numpy as np

from pkg.config.sample_data import visualize_files
from pkg.dataset.dataset import read_datasets
from pkg.gym_analyzer.train_lstm_dnn import train_lstm_dnn

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

model_path = "./model/pose_landmarker_heavy.task"
video_path = (
    "../../dataset/dataset_1/archive/barbell biceps curl/barbell biceps curl_44.mp4"
)
output_path = "./results/group.mp4"


if __name__ == "__main__":
    logging.info("log started")
    # visualize_data(visualize_files)

    x_train, x_test, label_processor = read_datasets("../../dataset/dataset_1/archive")
    train_lstm_dnn(x_train, x_test, label_processor)

    # train_rnn_keras(x_train, x_test, label_processor)
    # test_algorithms()
