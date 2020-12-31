import numpy as np
import argparse
import os
from tensorflow import keras
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FILE_EXTENSION = ".npy"


def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    plt.matshow(df_confusion)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def discrete_normalization(x):
    return int(10 * (int(x * 10.0) / 10.0 + 0.1)) / 10.0


def show_confusion_matrix(real_y, pred_y):
    real_res_y = pd.Series(
        np.array(list(map(discrete_normalization, np.reshape(real_y, (real_y.shape[0])))))
        , name="real"
    )
    pred_res_y = pd.Series(
        np.array(list(map(discrete_normalization, np.reshape(pred_y, (pred_y.shape[0])))))
        , name="pred"
    )
    confusion = pd.crosstab(real_res_y, pred_res_y)
    norm_confusion = confusion / confusion.sum(axis=1)
    plot_confusion_matrix(norm_confusion)


def save_model_files(model, model_path):
    model.save(model_path, save_format='h5', overwrite=True)


def extract_x_y_from_dataset(input_set):
    split = np.hsplit(input_set, [input_set.shape[1] - 1, input_set.shape[1]])
    return \
        np.reshape(split[0], (split[0].shape[0], split[0].shape[1], -1)), \
        np.reshape(split[1], (split[1].shape[0], split[1].shape[1], -1))


def create_model(show_summary: bool, n_features):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=30, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
    model.add(keras.layers.LSTM(units=30, return_sequences=True))
    model.add(keras.layers.LSTM(units=30, return_sequences=True))
    model.add(keras.layers.LSTM(units=30))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanSquaredError()]
    )

    if show_summary:
        print(model.summary())

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input folder with train.npy eval.npy and test.npy", default='../dataset')
    parser.add_argument('-o', '--output', help="output folder for models", default='../models')
    parser.add_argument('-l', '--logs', help="output folder for tensorboard logs", default='../logs/scalars/')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print("MyError: No input directory found.")
        return

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

    logs_dir = os.path.join(args.logs, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

    # model
    model = create_model(True, train_set_x.shape[1])

    # fit number_of_sets x number_of_features x 1
    model.fit(
        x=train_set_x,
        y=train_set_y,
        validation_data=(eval_set_x, eval_set_y),
        batch_size=512,
        epochs=100,
        callbacks=[tensorboard_callback]
    )

    model_save_path = os.path.join(args.output, "model" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_model_files(model, model_save_path)

    # model = keras.models.load_model("C:\\GIT\\models\\model")

    print("")
    print("Test")

    model.evaluate(
        x=test_set_x,
        y=test_set_y,
        batch_size=128
    )

    test_predict_y = \
        model.predict(
            x=test_set_x,
            batch_size=128)

    show_confusion_matrix(test_set_y, test_predict_y)


if __name__ == "__main__":
    main()
