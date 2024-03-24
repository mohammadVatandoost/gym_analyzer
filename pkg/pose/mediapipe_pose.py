import math
import cv2
import numpy as np
from time import time
import mediapipe as mp

# import mediapipe
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks.python.vision import PoseLandmarkerResult

from pkg.pose.skeleton import angle_connection
from pkg.video_reader.video_reader import VideoReader

import matplotlib.pyplot as plt

model_path = (
    "/home/mohammad/work/GYM/code/gym_analyzer/model/pose_landmarker_heavy.task"
)


class MediaPipePose:

    def __init__(self, video_reader: VideoReader = None) -> None:
        self.video_reader = video_reader
        # Initializing mediapipe pose class.
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        poseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        self.options = poseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
        )
        # self.pose = PoseLandmarker.create_from_options(self.options)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # Setting up the Pose function.
        # static_image_mode: if set to False, the detector is only invoked as needed, that is in the very first frame or when the tracker loses track. If set to True, the person detector is invoked on every input image.
        # smooth_landmarks – It is a boolean value that is if set to True, pose landmarks across different frames are filtered to reduce noise. But only works when static_image_mode is also set to False. Its default value is True.
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            # running_mode=VIDEO,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            smooth_landmarks=True,
        )

    def estimate_frame(self, frame, frame_timestamp_ms) -> PoseLandmarkerResult:
        # Perform pose detection after converting the image into RGB format.
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.pose.detect_for_video(mp_image, frame_timestamp_ms)
        # return self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def estimate_image(self, image):
        return self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def estimate(self):
        frame = self.video_reader.read_frame()
        # Perform pose detection after converting the image into RGB format.
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results
        # extract_keypoints function Check if any landmarks are found.
        # if results.pose_landmarks:
        #     return results
        # return None

    def draw_landmarks(self, image, results):
        """
        This function draws keypoints and landmarks detected by the human pose estimation model

        """

        pose_landmarks_list = results.pose_landmarks
        annotated_image = np.copy(image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                self.mp_pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image

    def flatten_key_points(self, result: PoseLandmarkerResult) -> np.array:
        """
        This function extracts the pose landmarks from the results object.
            Args:
                PoseLandmarkerResult: The results object returned by the Pose class.
            Returns:
                pose_keypoints: A list of pose landmarks.
        """
        pose = (
            np.array(
                [
                    [res.x, res.y, res.z, res.visibility]
                    for res in result.pose_landmarks[0]
                ]
            ).flatten()
            if result.pose_landmarks
            else np.zeros(33 * 4)
        )
        # self.plot_keypoints(PoseLandmarkerResult.pose_landmarks[0])
        return pose

    def plot_keypoints(self, pose_keypoints):
        """
        This function plots the pose landmarks on a 2D plot.
            Args:
                pose_keypoints: A list of pose landmarks.
        """
        x, y, labels = [], [], []
        for keypoint in pose_keypoints:
            x.append(keypoint.x)
            y.append(keypoint.y)
        # Create a figure and axis object.
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot the pose landmarks on the axis.
        ax.scatter(x, y, c="r", s=10)
        # Set the axis labels and title.
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Pose Landmarks")
        # Show the plot.
        plt.show()
        # Save the plot as a PNG image.
        # plt.savefig('pose_landmarks.png')
        # Clear the plot.
        plt.clf()
        # Close the plot.
        plt.close()
        return pose_keypoints

    def calculate_keypoint_angle(
        self, pose_key_points: list[landmark_pb2.NormalizedLandmark]
    ):
        angels = []
        for connection in angle_connection:
            angel = self.calculateAngle(
                pose_key_points[connection[0]],
                pose_key_points[connection[1]],
                pose_key_points[connection[2]],
            )
            angels.append(angel)

        return angels

    @staticmethod
    def calculateAngle(
        landmark1: landmark_pb2.NormalizedLandmark,
        landmark2: landmark_pb2.NormalizedLandmark,
        landmark3: landmark_pb2.NormalizedLandmark,
    ):
        """
        This function calculates angle between three different landmarks.
            Args:
                landmark1: The first landmark containing the x,y and z coordinates.
                landmark2: The second landmark containing the x,y and z coordinates.
                landmark3: The third landmark containing the x,y and z coordinates.
            Returns:
                angle: The calculated angle between the three landmarks.
        """
        # Calculate the angle between the three points
        angle = math.degrees(
            math.atan2(landmark3.y - landmark2.y, landmark3.x - landmark2.x)
            - math.atan2(landmark1.y - landmark2.y, landmark1.x - landmark2.x)
        )
        # Check if the angle is less than zero.
        if angle < 0:
            # Add 360 to the found angle.
            angle += 360
        # Return the calculated angle.
        return angle

    def reset(self):
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(self.options)
