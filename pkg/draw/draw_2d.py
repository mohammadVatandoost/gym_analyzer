from matplotlib import pyplot as plt


class Draw2d():

    def __init__(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.ax = ax
        self.fig = fig
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Pose Landmarks')