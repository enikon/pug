import numpy as np
import os
import argparse
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# def createBatches(df):
#     time = np.array(list(map(lambda d: (datetime.strptime(d, '%Y-%m-%d %H:%M')).hour, df.date.to_numpy())))
#     x = df.x.to_numpy()
#     y = df.y.to_numpy()
#
#     element_number = 0
#     global_row_number = 0
#     batches = []
#     batch = []
#
#     for row in time:
#         if element_number == 0:
#             batch = []
#         batch.append([time[global_row_number], x[global_row_number], y[global_row_number]])
#         global_row_number += 1
#         element_number += 1
#         if element_number == 24:
#             element_number = 0
#             batches.append(batch)
#
#     return batches

def parseData(df):
    time = np.array(list(map(lambda d: (datetime.strptime(d, '%Y-%m-%d %H:%M')).hour, df.date.to_numpy())))
    x = df.x.to_numpy()
    y = df.y.to_numpy()

    return time, x, y


def createBatches24(df):
    time, x_train, y_train = parseData(df)

    valPercentage = 10
    batchSize = 24

    element_number = 0
    global_row_number = 0
    batches_x = []
    batch_x = []
    batches_y = []
    batch_y = []

    for row in time:
        if element_number == 0:
            batch_x = []
            batch_y = []
        batch_x.append([x_train[global_row_number]])
        batch_y.append([y_train[global_row_number]])
        global_row_number += 1
        element_number += 1
        if element_number == batchSize:
            element_number = 0
            batches_x.append(batch_x)
            batches_y.append(batch_y)

    x_val = batches_x[0::10]
    y_val = batches_y[0::10]
    x_train = batches_x.copy()
    del x_train[valPercentage-1::valPercentage]
    y_train = batches_y.copy()
    del y_train[valPercentage-1::valPercentage]

    return x_train, y_train, x_val, y_val


def createBatches24concatenated(df):
    x_train, y_train, x_val, y_val = createBatches24(df)
    return np.concatenate(x_train), np.concatenate(y_train), np.concatenate(x_val), np.concatenate(y_val)


def create_rnn_model():
    model = keras.models.Sequential(
        keras.layers.SimpleRNN(
            units=1,
            batch_input_shape=(1, 24, 1),
            return_sequences=True,
            stateful=True
        )
    )

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer="adam",
        metrics=["accuracy"],
    )

    print(model.summary())

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input raw data", default='../raw_dataset')
    parser.add_argument('-o', '--output', help="output root folder", default='../dataset')
    parser.add_argument('-se', '--split_eval', help="percentage of data to be in evaluation set", default=0.1, type=float)
    parser.add_argument('-st', '--split_test', help="percentage of data to be in test set", default=0.1, type=float)

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print("MyError: No input directory found.")
        return

    # os.makedirs(args.output)

    df = pd.read_csv(
        filepath_or_buffer=args.input,
        sep=",",
        decimal=".",
        names=['date', 'x', 'y']
    )

    time, x_train, y_train = parseData(df)

    x_train, y_train, x_val, y_val = createBatches24(df)

    # print(time)
    # print(x)
    # print(y)

    print(x_train)
    print(y_train)
    print(x_val)
    print(y_val)

    model = create_rnn_model()

    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=1, epochs=40)

    # TODO
    # build RNN model
    # run RNN model


if __name__ == "__main__":
    main()

    # batch_size = 64
    # # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # # Each input sequence will be of size (28, 28) (height is treated like time).
    # input_dim = 28
    #
    # units = 64
    # output_size = 10  # labels are from 0 to 9
    #
    #
    # # Build the RNN model
    # def build_model(allow_cudnn_kernel=True):
    #     # CuDNN is only available at the layer level, and not at the cell level.
    #     # This means `LSTM(units)` will use the CuDNN kernel,
    #     # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    #     if allow_cudnn_kernel:
    #         # The LSTM layer with default options uses CuDNN.
    #         lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    #     else:
    #         # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    #         lstm_layer = keras.layers.RNN(
    #             keras.layers.LSTMCell(units), input_shape=(None, input_dim)
    #         )
    #     model = keras.models.Sequential(
    #         [
    #             lstm_layer,
    #             keras.layers.BatchNormalization(),
    #             keras.layers.Dense(output_size),
    #         ]
    #     )
    #     return model
    #
    #
    # mnist = keras.datasets.mnist
    #
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # sample, sample_label = x_train[0], y_train[0]
    #
    # model = build_model(allow_cudnn_kernel=True)
    #
    # model.compile(
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     optimizer="sgd",
    #     metrics=["accuracy"],
    # )
    #
    # print(model.summary())
    #
    # model.fit(
    #     x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
    # )
