import argparse
import numpy as np
import os
import tensorflow as tf
from itertools import accumulate
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

split_counts = [24, 7, 7, 24, 7, 5]
split_list = list(accumulate(split_counts[:-1]))


def create_model():
    #                               # batch size, num time steps, num features
    inputs_ch = tf.keras.Input(shape=(split_counts[0], 1))
    inputs_cd = tf.keras.Input(shape=(split_counts[1], 1))
    inputs_cw = tf.keras.Input(shape=(split_counts[2], 1))
    inputs_th = tf.keras.Input(shape=(split_counts[3], ))
    inputs_tw = tf.keras.Input(shape=(split_counts[4], ))
    inputs_tc = tf.keras.Input(shape=(split_counts[5], ))

    # Double CNN initial on sequential data

    cnn1h = tf.keras.layers.Conv1D(
        filters=5,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(inputs_ch)
    cnn1d = tf.keras.layers.Conv1D(
        filters=5,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(inputs_cd)
    cnn1w = tf.keras.layers.Conv1D(
        filters=5,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(inputs_cw)
    cnn2h = tf.keras.layers.Conv1D(
        filters=25,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(cnn1h)
    cnn2d = tf.keras.layers.Conv1D(
        filters=25,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(cnn1d)
    cnn2w = tf.keras.layers.Conv1D(
        filters=25,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(cnn1w)

    # RNN on sequential data

    lstm1h = tf.keras.layers.LSTM(
        units=10
    )(cnn2h)
    lstm1d = tf.keras.layers.LSTM(
        units=10
    )(cnn2d)
    lstm1w = tf.keras.layers.LSTM(
        units=10
    )(cnn2w)

    # Concatenate all
    concat0 = tf.keras.layers.Concatenate()([
        lstm1h,
        lstm1d,
        lstm1w,
        inputs_th,
        inputs_tw,
        inputs_tc
    ])

    # DNN Categorical
    dense1 = tf.keras.layers.Dense(40, activation=tf.nn.relu)(concat0)
    dense2 = tf.keras.layers.Dense(20, activation=tf.nn.relu)(dense1)
    outputs = tf.keras.layers.Dense(1, activation=None)(dense2)

    model = tf.keras.Model(
        inputs=[
            inputs_ch,
            inputs_cd,
            inputs_cw,
            inputs_th,
            inputs_tw,
            inputs_tc
        ], outputs=outputs)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input root folder", default='../dataset')
    parser.add_argument('-o', '--output', help="output root folder", default='../model')
    parser.add_argument('-l', '--logs', help="output folder for tensorboard logs", default='../logs/scalars/')

    # parser.add_argument('-qh', '--sequence_hours_size', help="length input per one entry", default=24, type=int)
    # parser.add_argument('-qd', '--sequence_days_size', help="how many days of the same hour in the input per one entry", default=7, type=int)
    # parser.add_argument('-qw', '--sequence_weekdays_size', help="how many days of the same hour and weekday in the input per one cycle", default=7, type=int)
    args = parser.parse_args()

    logs_dir = os.path.join(args.logs, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

    train_set = np.load(os.path.join(args.input, 'train.npy'), allow_pickle=True)
    eval_set = np.load(os.path.join(args.input, 'eval.npy'), allow_pickle=True)
    test_set = np.load(os.path.join(args.input, 'test.npy'), allow_pickle=True)
    train_weights_set = np.load(os.path.join(args.input, 'train_weights.npy'), allow_pickle=True)

    debug = True
    if debug:
        train_set = train_set[:100000]
        eval_set = eval_set[:100]
        test_set = test_set[:100]
        train_weights_set = train_weights_set[:100000]

    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError(),
        ]
    )
    model.fit(
        x=np.split(
            train_set[:, :-1],
            split_list,
            axis=1
        ),
        y=train_set[:, -1],
        validation_data=(
            np.split(
                eval_set[:, :-1],
                split_list,
                axis=1
            ),
            eval_set[:, -1]
        ),
        batch_size=128, epochs=200,
        verbose=1,
        sample_weight=train_weights_set,
        callbacks=[tensorboard_callback]
    )
    results = model.evaluate(
        np.split(
            test_set[:, :-1],
            split_list,
            axis=1
        ),
        test_set[:, -1],
        batch_size=128
    )

    h = 0
    h += 1


if __name__ == "__main__":
    main()
