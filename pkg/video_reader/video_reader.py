import cv2


reduced_frame_per_second = 8


class VideoReader:
    """Helper class for video utilities"""

    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        # self.cap.set(cv2.CAP_PROP_FPS, 2)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_counter = 0
        self._current_frame: cv2.typing.MatLike
        self.counter = 0
        self.frame_jump_target = 0

    def next_frame(self):
        self._current_frame = self.read_frame()
        if self._current_frame is None:
            return None
        return self._current_frame_counter

    def get_current_frame(self):
        """Get current frame of video being read"""
        return self._current_frame

    def read_frame(self):
        """Read a frame"""
        if self._current_frame_counter >= self._total_frames:
            return None
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is False or frame is None:
                return None
            self._current_frame_counter += 1
        else:
            return None

        if self.counter == 0:
            self.frame_jump_target = int(
                self.get_video_fps() / reduced_frame_per_second
            )
        self.counter += 1
        if self.counter < self.frame_jump_target:
            return self.read_frame()
        self.counter = 0
        return frame

    def read_n_frames(self, num_frames=1):
        """Read n frames"""
        frames_list = []
        for _ in range(num_frames):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret is False or frame is None:
                    return None
                frames_list.append(frame)
                self._current_frame_counter += 1
            else:
                return None
        return frames_list

    def is_opened(self):
        """Check is video capture is opened"""
        return self.cap.isOpened()

    def get_frame_width(self):
        """Get width of a frame"""
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_frame_height(self):
        """Get height of a frame"""
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame_timestamp(self):
        """Get Frames Timestamp"""
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def get_video_fps(self):
        """Get Frames per second of video"""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_current_frame(self):
        """Get current frame of video being read"""
        return self._current_frame

    def get_total_frames(self):
        """Get total frames of a video"""
        return self._total_frames

    def release(self):
        """Release video capture"""
        self.cap.release()

    def __del__(self):
        self.release()
