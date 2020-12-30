import numpy as np
import argparse
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

DATASET_FILE_EXTENSION = ".npy"


def extract_x_y_from_dataset(input_set):
    split = np.hsplit(input_set, [input_set.shape[1] - 1, input_set.shape[1]])
    return \
        np.reshape(split[0], (split[0].shape[0], split[0].shape[1], -1)), \
        np.reshape(split[1], (split[1].shape[0], split[1].shape[1], -1))


def create_model(show_summary: bool, n_features):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=30, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
    model.add(keras.layers.Conv1D(filters=30, kernel_size=3, activation='relu'))
    model.add(keras.layers.LSTM(units=20, return_sequences=True))
    model.add(keras.layers.LSTM(units=20, return_sequences=True))
    model.add(keras.layers.LSTM(units=20))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal'))

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer="adam",
        metrics=["accuracy"],
    )

    if show_summary:
        print(model.summary())

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input folder with train.npy eval.npy and test.npy", default='../dataset')
    parser.add_argument('-o', '--output', help="output folder for models", default='../model')
    parser.add_argument('-l', '--logs', help="output folder for tensorboard logs", default='../logs/scalars/')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print("MyError: No input directory found.")
        return

    logs_dir = os.path.join(args.logs, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

    train_set_path = os.path.join(args.input, 'train' + DATASET_FILE_EXTENSION)
    eval_set_path = os.path.join(args.input, 'eval' + DATASET_FILE_EXTENSION)
    test_set_path = os.path.join(args.input, 'test' + DATASET_FILE_EXTENSION)

    if not os.path.exists(train_set_path):
        print("MyError: File train.npy not found in input directory.")
    if not os.path.exists(eval_set_path):
        print("MyError: File eval.npy not found in input directory.")
    if not os.path.exists(test_set_path):
        print("MyError: File test.npy not found in input directory.")

    train_set = np.load(train_set_path, allow_pickle=True)
    eval_set = np.load(eval_set_path, allow_pickle=True)
    test_set = np.load(test_set_path, allow_pickle=True)

    train_set_x, train_set_y = extract_x_y_from_dataset(train_set)
    eval_set_x, eval_set_y = extract_x_y_from_dataset(eval_set)
    test_set_x, test_set_y = extract_x_y_from_dataset(test_set)

    # model
    model = create_model(True, train_set_x.shape[1])

    # fit number_of_sets x number_of_features x 1
    model.fit(
        x=train_set_x,
        y=train_set_y,
        validation_data=(eval_set_x, eval_set_y),
        batch_size=128,
        epochs=100,
        callbacks=[tensorboard_callback]
    )
    result = model.evaluate(
        x=test_set_x,
        y=test_set_y,
        batch_size=128
    )

    print(result)

if __name__ == "__main__":
    main()
