import logging
import os
import time
from keras.utils import to_categorical

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.models import Sequential, Model

from keras.layers import (LSTM, Dense, Concatenate, Attention, Dropout, Softmax,
                                     Input, Flatten, Activation, Bidirectional, Permute, multiply,
                                     ConvLSTM2D, MaxPooling3D, TimeDistributed, Conv2D, MaxPooling2D)

from pkg.dataset.dataset import ExerciseVideoData
from pkg.gym_analyzer.preprocess import preprocess_on_key_points
from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.pose.skeleton import angle_connection

# some hyperparamters
Batch_Size = 32
Max_Epochs = 500
# Num_Input_Values = 33*4  # 33 landmarks with 4 values (x, y, z, visibility)
Num_Input_Values = len(angle_connection)  # number of angle connection
FPS = 30
Sequence_Length = FPS * 1
Learning_Rate = 0.01


def train_lstm_dnn(x_train: list[ExerciseVideoData], x_test: list[ExerciseVideoData], label_processor):
    # Videos are going to be this many frames in length
    pose = MediaPipePose()
    # Path for exported data, numpy arrays
    data_path = os.path.join(os.getcwd(), 'keypoints')
    train_data, train_labels = preprocess_on_key_points(pose, x_train, label_processor, data_path, Sequence_Length)
    test_data, test_labels = preprocess_on_key_points(pose, x_test, label_processor, data_path, Sequence_Length)
    logging.info(f"Training data shape: {train_data.shape} and size: {len(train_data)}, Training labels data shape: {train_labels.shape} and size: {len(train_labels)}")
    logging.info(f"Testing data shape: {test_data.shape} and size: {len(test_data)}, Testing labels data shape: {test_labels.shape} and size: {len(test_labels)}")
    # Callbacks to be used during neural network training
    # Stop training when a monitored metric has stopped improving.
    es_callback = EarlyStopping(monitor='val_loss', min_delta=5e-4, patience=10, verbose=0, mode='min')
    # Reduce learning rate when a metric has stopped improving.
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=0, mode='min')
    chkpt_callback = ModelCheckpoint(filepath=data_path, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='min', save_freq=1)



    # Set up Tensorboard logging and callbacks
    NAME = f"ExerciseRecognition-LSTM-{int(time.time())}"
    log_dir = os.path.join(os.getcwd(), 'logs', NAME, '')
    tb_callback = TensorBoard(log_dir=log_dir)

    callbacks = [tb_callback, es_callback, lr_callback, chkpt_callback]

    lstm = build_lstm(label_processor)
    print(lstm.summary())
    lstm.fit(
        train_data,
        train_labels,
        batch_size=Batch_Size,
        epochs=Max_Epochs,
        validation_data=(test_data, test_labels),
        callbacks=callbacks
    )


def build_lstm(label_processor):
    lstm = Sequential()
    lstm.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(Sequence_Length, Num_Input_Values)))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(256, return_sequences=True, activation='relu'))
    lstm.add(LSTM(128, return_sequences=False, activation='relu'))
    lstm.add(Dense(128, activation='relu'))
    lstm.add(Dense(64, activation='relu'))
    lstm.add(Dense(len(label_processor.get_vocabulary()), activation='softmax'))

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=Learning_Rate)
    lstm.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return lstm