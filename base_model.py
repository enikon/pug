import numpy as np
import argparse
import os
from tensorflow import keras
from datetime import datetime
from plots import show_confusion_matrix
import troubleshooting

DATASET_FILE_EXTENSION = ".npy"
troubleshooting.tf_init()


def save_model_files(model, model_path):
    model.save(model_path, save_format='h5', overwrite=True)


def extract_x_y_from_dataset(input_set):
    split = np.hsplit(input_set, [input_set.shape[1] - 1, input_set.shape[1]])
    return \
        np.reshape(split[0], (split[0].shape[0], split[0].shape[1], -1)), \
        np.reshape(split[1], (split[1].shape[0], split[1].shape[1], -1))


def create_model(show_summary: bool, n_features):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=20, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
    model.add(keras.layers.LSTM(units=40, return_sequences=True))
    model.add(keras.layers.LSTM(units=40, return_sequences=True))
    model.add(keras.layers.LSTM(units=40))
    model.add(keras.layers.Dense(150, activation='relu'))
    model.add(keras.layers.Dense(75, activation='relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
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
    parser.add_argument('-c', '--checkpoints', help="output folder for checkpoint model", default='../checkpoints/')

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

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logs_dir = os.path.join(args.logs, datetime_str)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_dir)

    training_dir = os.path.normpath(
        os.path.join(
            args.checkpoints,
            datetime_str
        )
    )
    os.mkdir(training_dir)

    # required as ModelCheckpoint is using sys not lib
    checkpoint_filepath = os.path.join(
        training_dir,
        "saved-model-{epoch:03d}-{val_loss:.2f}.h5"
    )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_freq='epoch',
        period=5,
        verbose=1,
        monitor='val_loss'
    )

    # model
    model = create_model(True, train_set_x.shape[1])

    # fit number_of_sets x number_of_features x 1
    model.fit(
        x=train_set_x,
        y=train_set_y,
        validation_data=(eval_set_x, eval_set_y),
        batch_size=512,
        epochs=150,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )

    # model_save_path = os.path.join(args.output, "model" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # save_model_files(model, model_save_path)

    # model = keras.models.load_model("C:\\GIT\\models\\model20201231-182211")

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
