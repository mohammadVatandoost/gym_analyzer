import logging
import time

import cv2

from pkg.motion.motion import MotionDetection
from pkg.pose.openpose import OpenPose
from pkg.video_reader.video_reader import VideoReader


def visualize(video_path: str):
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
            logging.info(
                "frame is read, frame_count: %s, timestamp: %d",
                frame_count,
                sample_video_reader.get_frame_timestamp(),
            )
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    logging.info("frames is ended, releasing the resources")
    sample_video_reader.release()
    cv2.destroyAllWindows()
