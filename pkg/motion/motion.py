import cv2
import numpy as np

from pkg.video_reader.video_reader import VideoReader


class MotionDetection:
    def __init__(self, video_reader: VideoReader) -> None:
        self.video_reader = video_reader
        self.frame = video_reader.get_current_frame()

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parameters for Lucas Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Random colors for tracking
        self.color = np.random.randint(0, 255, (100, 3))

        self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.frame_gray , mask=None, **self.feature_params)


    def motion_detect(self):
        next_frame = self.video_reader.get_current_frame()
        diff = cv2.absdiff(self.frame, next_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def optical_flow(self):
        next_frame = self.video_reader.get_current_frame()
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.frame_gray, next_frame_gray,
            self.p0, None,
            **self.lk_params
        )

        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        self.frame_gray = next_frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        return good_old, good_new

    def marks_flow_vectors(self, frame, good_new, good_old):
        # Create a mask image for drawing purposes
        # ToDo: performance issue, each time creating new masl
        mask = np.zeros_like(frame)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        return img

    @staticmethod
    def marks_contours(frame, contours):
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 300:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
        # cv2.imshow("feed", frame)
