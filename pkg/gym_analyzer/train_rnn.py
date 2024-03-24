import logging
import keras
from pkg.dataset.dataset import ExerciseVideoData
from pkg.gym_analyzer.preprocess import build_feature_extractor, prepare_all_videos

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


# Utility for our sequence model.
def get_sequence_model(label_processor):
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def run_experiment(train_data, train_labels, test_data, test_labels):
    filepath = "/tmp/video_classifier/ckpt.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


# Utility for running experiments.
def run_experiment(train_data, train_labels, test_data, test_labels, label_processor):
    filepath = "../../data/weights/ckpt.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model(label_processor)
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    logging.info(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


def train_rnn_keras(
    x_train: list[ExerciseVideoData], x_test: list[ExerciseVideoData], label_processor
):
    feature_extractor = build_feature_extractor()
    train_data, train_labels = prepare_all_videos(
        feature_extractor, x_train, label_processor
    )
    test_data, test_labels = prepare_all_videos(
        feature_extractor, x_test, label_processor
    )
    logging.info(f"Frame features in train set: {train_data[0].shape}")
    logging.info(f"Frame masks in train set: {train_data[1].shape}")
    _, sequence_model = run_experiment(
        train_data, train_labels, test_data, test_labels, label_processor
    )


# def evaluate(x_train: list[ExerciseVideoData])
