import logging
import numpy as np

from pkg.pose.movenet_prediction import movenet_predication

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
# import mediapipe as mp
import cv2

from pkg.pose.movenet import Movenet
# from mediapipe.tasks import python
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
video_path = 'data/dataset/group/Open Group Workout Vol. I _ Hiit Training _ Circuit Exercises _ Tabata _ This is Fitness.mp4'
output_path = './results/group.mp4'

def read_video_file(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # Get the frame rate of the video
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    return cap, frame_rate


# def make_pose_landmarks():
#     BaseOptions = mp.tasks.BaseOptions
#     PoseLandmarker = mp.tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp.tasks.vision.RunningMode
#
#     # Create a pose landmarker instance with the video mode:
#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_path=model_path),
#         running_mode=VisionRunningMode.VIDEO)
#     return PoseLandmarker, options
#     # with PoseLandmarker.create_from_options(options) as landmarker:


# def draw_landmarks_on_image(rgb_image, detection_result):
#     pose_landmarks_list = detection_result.pose_landmarks
#     annotated_image = np.copy(rgb_image)
#
#     # Loop through the detected poses to visualize.
#     for idx in range(len(pose_landmarks_list)):
#         pose_landmarks = pose_landmarks_list[idx]
#
#         # Draw the pose landmarks.
#         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#         ])
#         solutions.drawing_utils.draw_landmarks(
#             annotated_image,
#             pose_landmarks_proto,
#             solutions.pose.POSE_CONNECTIONS,
#             solutions.drawing_styles.get_default_pose_landmarks_style())
#     return annotated_image


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     sample_video_reader = VideoReader(video_path)
#     # cap, frame_rate = read_video_file(video_path)
#     # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_width = sample_video_reader.get_frame_width()
#     # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     frame_height = sample_video_reader.get_frame_height()
#     # fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # out = cv2.VideoWriter(output_path, fps=frame_rate, frameSize=(frame_width, frame_height))
#
#     PoseLandmarker, options = make_pose_landmarks()
#     with PoseLandmarker.create_from_options(options) as landmarker:
#
#         while True:
#             # ret, frame = cap.read()
#             ret, frame = sample_video_reader.read_frame()
#             # Break the loop if we have reached the end of the video
#             if not ret:
#                 break
#
#             # Get the timestamp of the current frame
#             frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
#             # Convert the OpenCV frame to MediaPipe's Image object
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#             # image = mp.solutions.holistic.Holistic._to_image(frame)
#
#             # Process the frame with MediaPipe
#             # results = holistic.process(image)
#
#             # Perform pose landmarking on the provided single image.
#             # The pose landmarker must be created with the video mode.
#             pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))
#
#             # Access the landmarks and timestamp from the results
#             landmarks = pose_landmarker_result.pose_landmarks
#             # timestamp = pose_landmarker_result.timestamp
#             annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
#             # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#             #
#             # annotated_frame = frame.copy()
#             # landmarker.draw_landmarks(annotated_frame, landmarks)
#             # Write the annotated frame to the output video
#             # out.write(annotated_image)
#             cv2.imshow('Holistic Model', annotated_image)
#
#             # Exit the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     # Release the VideoCapture and close the OpenCV window
#     sample_video_reader.release()
#     # cap.release()
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.info("log started")
    # dense_motion(video_path, "gpu")
    sample_video_reader = VideoReader(video_path)
    frame_count = sample_video_reader.next_frame()
    time_stamp = sample_video_reader.get_frame_timestamp()
    motion_detector = MotionDetection(sample_video_reader)
    movenet = Movenet(sample_video_reader)
    movenet_predication(video_path)
    while frame_count is not None:
        logging.info("frame is read, frame_count: %s, timestamp: %d",
                     frame_count, sample_video_reader.get_frame_timestamp())
        points = movenet.estimate()
        try:
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

            # frame, bgr = motion_detector.dense_optical_flow_by_gpu()
            # cv2.imshow("feed", cv2.add(frame, bgr))

            points = movenet.estimate()
            # logging.info(f"points: {points}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        frame_count = sample_video_reader.next_frame()
        next_time_stamp = sample_video_reader.get_frame_timestamp()
        # 5 for computation delay
        # time.sleep((next_time_stamp-time_stamp-20)/ 1000)
        time_stamp = next_time_stamp

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logging.info("frames is ended, releasing the resources")
    sample_video_reader.release()
    cv2.destroyAllWindows()
