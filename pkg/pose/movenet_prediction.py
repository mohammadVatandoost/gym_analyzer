import logging
import time

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

import cv2
from pkg.pose.helper import draw_prediction_on_image, \
    init_crop_region, determine_crop_region, run_inference, draw_keypoints


# Download the model from TF Hub.
# model = hub.load(
#     "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
# movenet = model.signatures['serving_default']

def movenet_multiperson(input_image):
    # Load the input image.
    # image_path = 'PATH_TO_YOUR_IMAGE'
    # image = tf.io.read_file(image_path)
    # image = tf.compat.v1.image.decode_jpeg(image)
    input_image = tf.expand_dims(input_image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.cast(tf.image.resize_with_pad(input_image, 256, 256), dtype=tf.int32)

    # # Run model inference.
    # outputs = movenet(input_image)
    # # Output is a [1, 6, 56] tensor.
    # logging.info(f"movenet_multiperson outputs['output_0']={outputs['output_0']}")
    # keypoints = outputs['output_0']
    logging.info("movenet_multiperson")
    return  input_image



def predict_movenet_for_video(video_path):
    model_name = "movenet_lightning"
    # interpreter = tf.lite.Interpreter(model_path="model/lite-model_movenet_singlepose_lightning_3.tflite")
    # interpreter = tf.lite.Interpreter(model_path="model/movenet_thunder_f16.tflite")
    interpreter = tf.lite.Interpreter(model_path="model/movenet_thunder_f16.tflite")
    # interpreter = tf.lite.Interpreter(model_path="model/movenet_multipose_multinet.pb")
    input_size = 192
    thunder_input_size = 256
    input_size = thunder_input_size

    interpreter.allocate_tensors()

    def movenet(input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        # input_image = tf.cast(input_image, dtype=tf.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        logging.info(f"movenet output_details: {output_details}")
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        return keypoints_with_scores



    # Load the input video file.
    cap = cv2.VideoCapture(video_path)

    # Initialize the frame count
    frame_count = 0

    output_images = []
    output_keypoints = []

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            image_height, image_width, _ = frame.shape

            # Initialize only during the first frame
            if frame_count == 0:
                crop_region = init_crop_region(image_height, image_width)

            # movenet_multiperson(frame)
            # continue
            # Crop and resize according to model input and then return the keypoint with scores
            keypoints_with_scores = run_inference(
                movenet, frame, crop_region,
                crop_size=[input_size, input_size])



            output_keypoints.append(keypoints_with_scores)

            # For GIF Visualization
            # output_images.append(draw_prediction_on_image(
            #     frame.astype(np.int32),
            #     keypoints_with_scores, crop_region=None,
            #     close_figure=True, output_image_height=300))
            # logging.info(f"keypoints is found, {keypoints_with_scores}")
            # result = draw_prediction_on_image(
            #     frame.astype(np.int32),
            #     keypoints_with_scores, crop_region=None,
            #     close_figure=True, output_image_height=300)
            result = draw_keypoints(frame, keypoints_with_scores)
            # logging.info(f"result: {result}")
            cv2.imshow("feed", result)
            time.sleep(0.03)
            # Crops the image for model
            crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

            # output = np.stack(output_images, axis=0)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if ret != True:
            break

    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # will be stored as animation.gif
    # to_gif(output, fps=10)

    print("Frame count : ", frame_count)

    return output_keypoints


def movenet_predication(video_path):

    # video_path = os.path.join('video', 'faceon.mp4')
    output_keypoints = predict_movenet_for_video(video_path)

    if output_keypoints is not None:
        print("Converted to Gif Successfully")


