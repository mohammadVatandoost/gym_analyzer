import logging

import cv2

from pkg.draw.draw import Draw
from pkg.video_reader.video_reader import VideoReader


class Pose():
    """ Base: Pose Class """
    def __init__(self, video_reader: VideoReader) -> None:
        self.video_reader = video_reader
        self.draw = Draw()

    def estimate(self) -> (bool, cv2.typing.MatLike):
        """ Estimate pose (base function) """
        if self.video_reader.is_opened() is False:
            logging.error("video reader is not open")
            return False, None

        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), self.video_reader.get_video_fps(),
                              (self.video_reader.get_frame_width(),
                               self.video_reader.get_frame_height()))
        # while self.video_reader.is_opened():
        image = self.video_reader.read_frame()
        if image is None:
            logging.error("Ignoring empty camera frame.")
            return False, None

        frame_timestamp_ms = self.video_reader.get_video_fps()
        # Convert the OpenCV frame to MediaPipe's Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # image = mp.solutions.holistic.Holistic._to_image(frame)

        # Process the frame with MediaPipe
        # results = holistic.process(image)

        # Perform pose landmarking on the provided single image.
        # The pose landmarker must be created with the video mode.
        pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp_ms))

        # Access the landmarks and timestamp from the results
        landmarks = pose_landmarker_result.pose_landmarks
        # timestamp = pose_landmarker_result.timestamp
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        #
        # annotated_frame = frame.copy()
        # landmarker.draw_landmarks(annotated_frame, landmarks)
        # Write the annotated frame to the output video
        # out.write(annotated_image)
        cv2.imshow('Holistic Model', annotated_image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # self.video_reader.release()