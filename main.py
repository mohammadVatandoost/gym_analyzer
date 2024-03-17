import logging
import numpy as np

from pkg.config.sample_data import visualize_files
from pkg.dataset.dataset import read_datasets
from pkg.gym_analyzer.train_lstm_dnn import train_lstm_dnn
from pkg.gym_analyzer.train_rnn import train_rnn_keras
from pkg.gym_analyzer.visualize import visualize_data
from pkg.pose.openpose import OpenPose

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import time
import cv2

from pkg.video_reader.video_reader import VideoReader
from pkg.motion.motion import MotionDetection
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2



logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)
# import tensorflow as tf
#
#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Initialize MediaPipe
# mp_holistic = mp.solutions.holistic

model_path = './model/pose_landmarker_heavy.task'
# video_path = './dataset/pushups/video_2023-10-07_09-37-41.mp4'
# video_path = './dataset/pushups/video_2023-10-08_17-25-11.mp4'
# video_path = './dataset/squat/video_2023-10-08_17-25-31.mp4'
# video_path = 'data/dataset/combine/combine_single.mp4'
# video_path = 'data/dataset/group/group.mp4'
video_path = '../../dataset/dataset_1/archive/barbell biceps curl/barbell biceps curl_44.mp4'
output_path = './results/group.mp4'

# def read_video_file(file_path):
#     cap = cv2.VideoCapture(file_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         exit()
#     # Get the frame rate of the video
#     frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#     return cap, frame_rate


if __name__ == '__main__':
    logging.info("log started")
    visualize_data(visualize_files)

    # x_train, x_test, label_processor = read_datasets("../../dataset/dataset_1/archive")
    # train_lstm_dnn(x_train, x_test, label_processor)

    # train_rnn_keras(x_train, x_test, label_processor)
    # test_algorithms()


def test_algorithms():
    # dense_motion(video_path, "gpu")
    sample_video_reader = VideoReader(video_path)
    frame_count = sample_video_reader.next_frame()
    time_stamp = sample_video_reader.get_frame_timestamp()
    motion_detector = MotionDetection(sample_video_reader)
    # movenet = Movenet(sample_video_reader)
    # movenet_predication(video_path)
    # pose_estimation_by_openpose(video_path)
    open_pose = OpenPose(sample_video_reader)
    while frame_count is not None:

        # points = movenet.estimate()
        try:
            logging.info("frame is read, frame_count: %s, timestamp: %d",
                         frame_count, sample_video_reader.get_frame_timestamp())
            # contours = motion_detector.motion_detect()
            # frame = motion_detector.marks_contours(sample_video_reader.get_current_frame(), contours)

            # good_old, good_new, err = motion_detector.optical_flow_Lucas_Kanade()
            # if err is not None:
            #     logging.error(err)
            #     frame_count = sample_video_reader.next_frame()
            #     next_time_stamp = sample_video_reader.get_frame_timestamp()
            #     time.sleep((next_time_stamp - time_stamp) / 1000)
            #     time_stamp = next_time_stamp
            #     continue
            # frame = motion_detector.marks_flow_vectors(
            #     sample_video_reader.get_current_frame(),
            #     good_new,
            #     good_old
            # )

            # bgr = motion_detector.optical_flow_dense()

            # bgr = motion_detector.optical_flow_nvidia()
            # frame = cv2.add(sample_video_reader.get_current_frame(), bgr)

            # frame, bgr = motion_detector.dense_optical_flow_by_gpu(isBGR=True)
            # cv2.imshow("feed", cv2.add(frame, bgr))

            frame, contours = motion_detector.dense_optical_flow_by_gpu()
            cv2.imshow("motion", motion_detector.draw_contours(frame, contours))

            result = open_pose.estimate()
            cv2.imshow("pose", result)
            # cv2.imshow("feed", motion_detector.draw_contours(result, contours))

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        frame_count = sample_video_reader.next_frame()
        next_time_stamp = sample_video_reader.get_frame_timestamp()
        # 5 for computation delay
        # time.sleep((next_time_stamp-time_stamp-20)/ 1000)
        time.sleep(0.060)
        time_stamp = next_time_stamp

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logging.info("frames is ended, releasing the resources")
    sample_video_reader.release()
    cv2.destroyAllWindows()
