from matplotlib import pyplot as plt
import numpy as np


class Draw2d:

    def __init__(self, title):
        plt.ion()  # Turn on interactive plotting
        fig, ax = plt.subplots(figsize=(6, 10))
        self.ax = ax
        self.fig = fig
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(title)

        # Define the important skeleton structure based on the provided keypoints
        self.skeleton = [
            (0, 1),
            (1, 2),
            (2, 3),  # Left eye
            (0, 4),
            (4, 5),
            (5, 6),  # Right eye
            (1, 7),
            (4, 8),  # Ears to eyes
            (9, 10),  # Mouth
            (11, 13),
            (13, 15),
            (15, 17),
            (15, 19),
            (15, 21),  # Left arm + fingers
            (12, 14),
            (14, 16),
            (16, 18),
            (16, 20),
            (16, 22),  # Right arm + fingers
            (11, 23),
            (12, 24),  # Shoulders to hips
            (23, 25),
            (25, 27),
            (27, 29),
            (27, 31),  # Left leg + foot
            (24, 26),
            (26, 28),
            (28, 30),
            (28, 32),  # Right leg + foot
            (11, 12),  # Shoulder connection
            (23, 24),  # Hip connection
        ]

    def plot_keypoints(self, pose_keypoints):
        x, y, labels = [], [], []
        for keypoint in pose_keypoints:
            x.append(keypoint.x)
            y.append(1 - keypoint.y)
        self.ax.scatter(x, y, c="r", s=10)
        for connection in self.skeleton:
            start_point = pose_keypoints[connection[0]]
            end_point = pose_keypoints[connection[1]]
            self.ax.plot(
                [start_point.x, end_point.x], [1 - start_point.y, 1 - end_point.y], "r-"
            )
        self.fig.canvas.draw()

    def imshow(self, image):
        self.ax.imshow(image)
        self.fig.canvas.draw()

    def get_image(self):
        return np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")

    def plot_show(self):
        plt.pause(0.001)
        plt.show(block=False)

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Pose Landmarks")

    def close_plot(self):
        plt.clf()
        plt.close()
