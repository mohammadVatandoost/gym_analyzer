# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

from pkg.video_reader.video_reader import VideoReader


class Movenet():
    def __init__(self, video_reader: VideoReader) -> None:
        self.video_reader = video_reader
        self.model = hub.load(
            "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
        # Download the model from TF Hub.
        self.movenet = self.model.signatures['serving_default']

    def estimate(self):
        frame = self.video_reader.get_current_frame()
        resized_frame = tf.cast(tf.image.resize_with_pad(frame, 256, 256), dtype=tf.int32)
        # Run model inference.
        outputs = self.movenet(resized_frame)

        # Output is a [1, 6, 56] tensor.
        keypoints = outputs['output_0']

# # Load the input image.
# image_path = 'PATH_TO_YOUR_IMAGE'
# image = tf.io.read_file(image_path)
# image = tf.compat.v1.image.decode_jpeg(image)
# image = tf.expand_dims(image, axis=0)
# # Resize and pad the image to keep the aspect ratio and fit the expected size.



