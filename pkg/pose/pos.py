import cv2

from pkg.video_reader.video_reader import VideoReader


class Pose():
    """ Base: Pose Class """
    def __init__(self, video_reader: VideoReader) -> None:
        self.video_reader = video_reader
        self.draw = Draw(self.video_reader.get_frame_width(), self.video_reader.get_frame_height())

    def estimate(self) -> None:
        """ Estimate pose (base function) """
        if self.video_reader.is_opened() is False:
            print("Error File Not Found.")

        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), self.video_reader.get_video_fps(),
                              (self.video_reader.get_frame_width(),
                               self.video_reader.get_frame_height()))
        while self.video_reader.is_opened():
            image = self.video_reader.read_frame()
            if image is None:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = self.draw.overlay(image)
            image = self.draw.skeleton(image, results)

            if results.pose_landmarks is not None:
                self.key_points = self.get_keypoints(image, results)
                estimated_pose = self.predict_pose()
                if estimated_pose is not None:
                    self.current_pose = estimated_pose
                    self.pose_tracker.append(self.current_pose)
                    if len(self.pose_tracker) == 10 and len(set(self.pose_tracker[-6:])) == 1:
                        image = self.draw.pose_text(image, "Prediction: " + estimated_pose)

            if len(self.pose_tracker) == 10:
                del self.pose_tracker[0]
                self.prev_pose = self.pose_tracker[-1]

            out.write(image)
            cv2.imshow('Estimation of Exercise', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_reader.release()