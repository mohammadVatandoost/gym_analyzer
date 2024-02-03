import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

from pkg.video_reader.video_reader import VideoReader


class MediaPipePose():

    def __init__(self, video_reader: VideoReader = None) -> None:
        self.video_reader = video_reader
        # Initializing mediapipe pose class.
        self.mp_pose = mp.solutions.pose
        # Setting up the Pose function.
        # static_image_mode: if set to False, the detector is only invoked as needed, that is in the very first frame or when the tracker loses track. If set to True, the person detector is invoked on every input image.
        # smooth_landmarks – It is a boolean value that is if set to True, pose landmarks across different frames are filtered to reduce noise. But only works when static_image_mode is also set to False. Its default value is True.
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
            model_complexity=2,
            smooth_landmarks=True
        )
        # Initializing mediapipe drawing class, useful for annotation.
        self.mp_drawing = mp.solutions.drawing_utils

    def estimate_frame(self, frame):
        # Perform pose detection after converting the image into RGB format.
        return self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def estimate(self):
        frame = self.video_reader.read_frame()
        # Perform pose detection after converting the image into RGB format.
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results
        # extract_keypoints function Check if any landmarks are found.
        # if results.pose_landmarks:
        #     return results
        # return None


    def extract_keypoints(self, results):
        '''
        This function extracts the pose landmarks from the results object.
            Args:
                results: The results object returned by the Pose class.
            Returns:
                pose_keypoints: A list of pose landmarks.
        '''
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return pose


    def calculateAngle(self, landmark1, landmark2, landmark3):
        '''
        This function calculates angle between three different landmarks.
            Args:
                landmark1: The first landmark containing the x,y and z coordinates.
                landmark2: The second landmark containing the x,y and z coordinates.
                landmark3: The third landmark containing the x,y and z coordinates.
            Returns:
                angle: The calculated angle between the three landmarks.
        '''
        # Get the required landmarks coordinates.
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3
        # Calculate the angle between the three points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # Check if the angle is less than zero.
        if angle < 0:
            # Add 360 to the found angle.
            angle += 360
        # Return the calculated angle.
        return angle
