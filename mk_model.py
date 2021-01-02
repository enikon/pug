import argparse
import numpy as np
import os
import math
import tensorflow as tf
from itertools import accumulate
from datetime import datetime
from plots import show_confusion_matrix
import troubleshooting
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

troubleshooting.tf_init()

split_counts = [24, 7, 7, 24, 7, 5]
split_list = list(accumulate(split_counts[:-1]))


def create_model():
    #                    # batch size, num time steps, num features
    inputs_ch = tf.keras.Input(shape=(split_counts[0], 1))
    inputs_cd = tf.keras.Input(shape=(split_counts[1], 1))
    inputs_cw = tf.keras.Input(shape=(split_counts[2], 1))
    inputs_th = tf.keras.Input(shape=(split_counts[3],))
    inputs_tw = tf.keras.Input(shape=(split_counts[4],))
    inputs_tc = tf.keras.Input(shape=(split_counts[5],))

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
        filters=5,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(cnn1d)
    cnn2w = tf.keras.layers.Conv1D(
        filters=5,
        kernel_size=3,
        activation='relu',
        padding='same'
    )(cnn1w)

    # RNN on sequential data

    lstm1h = tf.keras.layers.LSTM(
        units=5
    )(cnn2h)
    lstm1d = tf.keras.layers.LSTM(
        units=5
    )(cnn2d)
    lstm1w = tf.keras.layers.LSTM(
        units=5
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
    dense1 = tf.keras.layers.Dense(15, activation=tf.nn.relu)(concat0)
    dense2 = tf.keras.layers.Dense(10, activation=tf.nn.relu)(dense1)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense2)

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
    BATCH_SIZE = 64
    EPOCHS = 20
    EPOCH_SAVE_PERIOD = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input root folder", default='../dataset')
    parser.add_argument('-o', '--output', help="output root folder", default='../model')
    parser.add_argument('-l', '--logs', help="output folder for tensorboard logs", default='../logs/scalars/')
    parser.add_argument('-c', '--checkpoints', help="output folder for checkpoint model", default='../checkpoints/')

    # parser.add_argument('-qh', '--sequence_hours_size', help="length input per one entry", default=24, type=int)
    # parser.add_argument('-qd', '--sequence_days_size', help="how many days of the same hour in the input per one entry", default=7, type=int)
    # parser.add_argument('-qw', '--sequence_weekdays_size', help="how many days of the same hour and weekday in the input per one cycle", default=7, type=int)
    args = parser.parse_args()

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.

    train_set = np.load(os.path.join(args.input, 'train.npy'), allow_pickle=True)
    eval_set = np.load(os.path.join(args.input, 'eval.npy'), allow_pickle=True)
    test_set = np.load(os.path.join(args.input, 'test.npy'), allow_pickle=True)
    train_weights_set = np.load(os.path.join(args.input, 'train_weights.npy'), allow_pickle=True)
    eval_weights_set = np.load(os.path.join(args.input, 'eval_weights.npy'), allow_pickle=True)

    debug = True
    scale = 0.01
    if debug:
        train_set = train_set[:round(len(train_set) * scale)]
        eval_set = eval_set[:round(len(eval_set) * scale)]
        test_set = test_set[:round(len(test_set) * scale)]
        train_weights_set = train_weights_set[:round(len(train_weights_set) * scale)]
        eval_weights_set = eval_weights_set[:round(len(eval_weights_set) * scale)]

    # Callbacks

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logs_dir = os.path.join(args.logs, datetime_str)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

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

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_freq='epoch', #EPOCH_SAVE_PERIOD * (int(math.ceil(len(train_set) / BATCH_SIZE))) + 1,
        period=EPOCH_SAVE_PERIOD,
        verbose=1,
        monitor='val_loss'
    )

    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
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
        sample_weight=train_weights_set,
        validation_data=(
            np.split(
                eval_set[:, :-1],
                split_list,
                axis=1
            ),
            eval_set[:, -1],
            eval_weights_set
        ),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        verbose=1,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )
    results = model.evaluate(
        np.split(
            test_set[:, :-1],
            split_list,
            axis=1
        ),
        test_set[:, -1],
        batch_size=BATCH_SIZE
    )

    test_predict = model.predict(
        np.split(
            test_set[:, :-1],
            split_list,
            axis=1
        ),
        batch_size=BATCH_SIZE
    )

    show_confusion_matrix(
        np.expand_dims(np.expand_dims(test_set[:, -1], -1), -1),
        test_predict
    )

    h = 0
    h += 1


if __name__ == "__main__":
    main()