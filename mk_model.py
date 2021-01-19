import argparse
import numpy as np
import os
import math
import tensorflow as tf
from itertools import accumulate
from datetime import datetime
from sklearn.metrics import f1_score
import tensorflow_addons as tfa

import utils
from plots import show_confusion_matrix, show_confusion_matrix_classes
import troubleshooting

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

troubleshooting.tf_init()

split_counts = [24, 7, 7, 24, 7, 5]


def main(_args):

    BATCH_SIZE = 512
    EPOCHS = 50
    EPOCH_SAVE_PERIOD = 5
    DATASET_PERCENTAGE = 0.001

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input root folder", default='../dataset_classes')
    parser.add_argument('-o', '--output', help="output root folder", default='../model')
    parser.add_argument('-l', '--logs', help="output folder for tensorboard logs", default='../logs/scalars/')
    parser.add_argument('-c', '--checkpoints', help="output folder for checkpoint model", default='../checkpoints/')

    parser.add_argument('-cn', '--classes_number', help="number of output classes", default=5, type=int)
    parser.add_argument('-tt', '--train_type', help="name of train file", default='train')
    parser.add_argument('-et', '--eval_type', help="name of eval file", default='eval')
    parser.add_argument('-st', '--test_type', help="name of test file", default='test')
    parser.add_argument('-iw', '--is_weighted', help="will training use sample weights", default=True, type=utils.str2bool, nargs='?', const=True)

    parser.add_argument('-m', '--model_name', help="name of model script in script_models directory", default="new_baseline")
    parser.add_argument('-qh', '--sequence_hours_size', help="length input per one entry", default=24, type=int)
    parser.add_argument('-qd', '--sequence_days_size', help="how many days of the same hour in the input per one entry", default=7, type=int)
    parser.add_argument('-qw', '--sequence_weekdays_size', help="how many days of the same hour and weekday in the input per one cycle", default=7, type=int)
    args = parser.parse_args(*_args)

    # Required for qh qd qw changes
    split_counts[0] = args.sequence_hours_size
    split_counts[1] = args.sequence_days_size
    split_counts[2] = args.sequence_weekdays_size
    split_list = list(accumulate(split_counts[:-1]))

    model_script = __import__("script_models."+args.model_name, fromlist=[''])

    train_set = np.load(os.path.join(args.input, args.train_type+'.npy'), allow_pickle=True)
    eval_set = np.load(os.path.join(args.input, args.eval_type + '.npy'), allow_pickle=True)
    test_set = np.load(os.path.join(args.input, args.test_type + '.npy'), allow_pickle=True)
    train_weights_set = None
    eval_weights_set = None

    if args.is_weighted:
        train_weights_set = np.load(os.path.join(args.input, 'train_weights.npy'), allow_pickle=True)
        eval_weights_set = np.load(os.path.join(args.input, 'eval_weights.npy'), allow_pickle=True)

    debug = False
    scale = DATASET_PERCENTAGE
    if debug:
        train_set = train_set[:round(len(train_set) * scale)]
        eval_set = eval_set[:round(len(eval_set) * scale)]
        test_set = test_set[:round(len(test_set) * scale)]
        if args.is_weighted:
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
        "saved-model-{epoch:03d}.h5"
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_freq=EPOCH_SAVE_PERIOD * (int(math.ceil(len(train_set) / BATCH_SIZE))),
        #period=EPOCH_SAVE_PERIOD,
        verbose=1,
        monitor='val_loss'
    )

    learning_rate = 1e-4
    model = model_script.create_model(args.classes_number, split_counts)
    if args.classes_number == 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
            ]
        )
    elif args.classes_number == 2:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Hinge(),
                tfa.metrics.F1Score(num_classes=1, threshold=0.5)
            ]
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(),
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.CategoricalHinge(),
                tfa.metrics.F1Score(num_classes=args.classes_number)
            ]
        )

    x_slice = slice(0, -args.classes_number)
    y_slice = slice(-args.classes_number, None)

    if args.classes_number == 0 or args.classes_number == 2:
        x_slice = slice(0, -1)
        y_slice = slice(-1, None)

    # model = tf.keras.models.load_model("C:\\GIT\\checkpoints\\20210110-150053\\saved-model-025-1.02.h5")

    model.fit(
        x=np.split(
            train_set[:, x_slice],
            split_list,
            axis=1
        ),
        y=train_set[:, y_slice],
        sample_weight=train_weights_set,
        validation_data=(
            np.split(
                eval_set[:, x_slice],
                split_list,
                axis=1
            ),
            eval_set[:, y_slice],
            eval_weights_set
        ),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        verbose=2,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )

    results = model.evaluate(
        np.split(
            test_set[:, x_slice],
            split_list,
            axis=1
        ),
        test_set[:, y_slice],
        batch_size=BATCH_SIZE
    )

    test_predict = model.predict(
        np.split(
            test_set[:, x_slice],
            split_list,
            axis=1
        ),
        batch_size=BATCH_SIZE
    )

    if args.classes_number == 2:
        test_pred = np.array(list(map(lambda tab: [1.] if tab[0] > 0.5 else [0.], test_predict)))
    else:
        test_pred = test_predict

    show_confusion_matrix_classes(
        test_set[:, y_slice],
        test_pred,
        args.classes_number
    )

    if args.classes_number == 2:
        selected_real = np.array(list(map(lambda tab: int(tab[0]), test_set[:, y_slice])))
        selected_pred = np.array(list(map(lambda tab: int(tab[0]), test_pred)))
    else:
        selected_real = np.array(list(map(lambda tab: np.argmax(tab), test_set[:, y_slice])))
        selected_pred = np.array(list(map(lambda tab: np.argmax(tab), test_pred)))

    f1 = f1_score(selected_real, selected_pred, average='macro')

    print("F1_score_test: ")
    print(f1)

    h = 0
    h += 1


if __name__ == "__main__":
    main({})
