import pandas as pd
import numpy as np
import os
import argparse
import shutil

from tensorflow.python.keras.utils.np_utils import to_categorical


def cycle_array(source, sequence_length, offset, array_range):
    """
    sequence_length -> how many values will there be in this sequence per one entry
    offset -> how far apart are values in the sequence
    """
    dataset_size = array_range[1] - array_range[0]
    target = np.zeros((dataset_size, sequence_length))
    for i in range(sequence_length):
        tmp_beg = array_range[0] - (sequence_length - 1) * offset
        tmp_end = array_range[1] - (sequence_length - 1) * offset
        target[:, i] = source[tmp_beg+i*offset: tmp_end+i*offset]
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input root folder", default='../raw_dataset')
    parser.add_argument('-o', '--output', help="output root folder", default='../dataset')
    parser.add_argument('-seu', '--split_eval_user', help="percentage of data to be in evaluation set by users", default=0.0, type=float)
    parser.add_argument('-stu', '--split_test_user', help="percentage of data to be in test set by users", default=0.0, type=float)
    parser.add_argument('-set', '--split_eval_time', help="percentage of data to be in evaluation set by time period", default=0.1, type=float)
    parser.add_argument('-stt', '--split_test_time', help="percentage of data to be in test set by time period", default=0.1, type=float)

    parser.add_argument('-qh', '--sequence_hours_size', help="length input per one entry", default=24, type=int)
    parser.add_argument('-qd', '--sequence_days_size', help="how many days of the same hour in the input per one entry", default=7, type=int)
    parser.add_argument('-qw', '--sequence_weekdays_size', help="how many days of the same hour and weekday in the input per one cycle", default=7, type=int)

    parser.add_argument('-cn', '--classes_number', help="how many classes in classification, 0 means regression",
                        default=0, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print("MyError: No input directory found.")
        return

    # Clear directory
    # TODO DONE WARNING failsafe
    if os.path.realpath(args.output) == "C:\\GIT\\dataset" \
            and os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    train_set_path = os.path.join(args.output, 'train')
    eval_set_path = os.path.join(args.output, 'eval')
    test_set_path = os.path.join(args.output, 'test')

    print('START')

    train_set = []
    eval_set = []
    test_set = []

    for input_dirs in os.listdir(args.input):
        input_cat_dir = os.path.join(args.input, input_dirs)

        input_files = os.listdir(input_cat_dir)
        input_files_index = -1
        input_files_size = len(input_files)
        users_size_eval = int(input_files_size * args.split_eval_user)
        users_size_test = int(input_files_size * args.split_test_user)
        users_size_train = input_files_size - users_size_eval - users_size_test

        debug_log_threshold = 10.0

        cat_number = -1 + int(''.join(filter(str.isdigit, input_cat_dir)))
        if cat_number != 3:
            continue

        for input_file in input_files:
            input_files_index += 1

            if input_files_index*100.0/input_files_size > debug_log_threshold:
                print(debug_log_threshold, "%")
                debug_log_threshold += 10.0

            df = pd.read_csv(
                filepath_or_buffer=os.path.join(input_cat_dir, input_file),
                sep=",",
                decimal=".",
                names=["date", "performance", "loss"],
                dtype={"date": "str", "performance": "float", "loss": "float"},
                parse_dates=["date"]
            )
            df = df.assign(weekday=df['date'].dt.dayofweek)
            df = df.assign(hour=df['date'].dt.hour)
            df = df.assign(category=cat_number)

            # ---------------------
            #  SEQUENCE EXTRACTION
            # ---------------------

            #TODO co zrobic z utratami jak sa w inpucie ?
            arr_base = df["performance"].to_numpy()

            cycle_range = [
                (args.sequence_weekdays_size-1) * 24 * 7,
                len(arr_base)
            ]
            dataset_size = cycle_range[1] - cycle_range[0]

            arr_ch = cycle_array(
                arr_base,
                args.sequence_hours_size, 1,
                cycle_range)
            arr_cd = cycle_array(
                arr_base,
                args.sequence_days_size, 24,
                cycle_range)
            arr_cw = cycle_array(
                arr_base,
                args.sequence_weekdays_size, 24*7,
                cycle_range)

            df_trimmed = df.iloc[cycle_range[0]:cycle_range[1]]
            arr_cat = np.zeros((dataset_size, 5))
            arr_cat[:, cat_number] = 1.0

            arr_weekday = np.eye(7)[df_trimmed["weekday"].to_numpy()]
            arr_hour = np.eye(24)[df_trimmed["hour"].to_numpy()]

            arr_loss = np.expand_dims(df_trimmed["loss"].to_numpy(), axis=-1)

            # TODO Fuzzify class assignment for border classes

            if args.classes_number != 0:
                arr_loss = to_categorical(
                    np.round(arr_loss * (args.classes_number-1)).astype(float),
                    num_classes=args.classes_number
                )

            arr_stacked = np.hstack((
                arr_ch, arr_cd, arr_cw,
                arr_hour, arr_weekday,
                arr_cat,
                arr_loss
             ))

            # -----------------------
            #  TRAIN/EVAL/TEST SPLIT
            # -----------------------

            dataset_size_eval = int(dataset_size * args.split_eval_time)
            dataset_size_test = int(dataset_size * args.split_test_time)
            dataset_size_train = dataset_size - dataset_size_eval - dataset_size_test

            dataset_slice_train = slice(0, dataset_size_train)
            dataset_slice_eval = slice(dataset_slice_train.stop, dataset_slice_train.stop + dataset_size_eval)
            dataset_slice_test = slice(dataset_slice_eval.stop, dataset_slice_eval.stop + dataset_size_test)

            if input_files_index < users_size_train:
                train_set.extend(arr_stacked[dataset_slice_train, :])
                eval_set.extend(arr_stacked[dataset_slice_eval, :])
                test_set.extend(arr_stacked[dataset_slice_test, :])
            elif input_files_index < users_size_train + users_size_eval:
                eval_set.extend(arr_stacked)
            else:
                test_set.extend(arr_stacked)

        print("100.0 %")

    np.save(train_set_path, np.stack(train_set), allow_pickle=True)
    np.save(eval_set_path, np.stack(eval_set), allow_pickle=True)
    np.save(test_set_path, np.stack(test_set), allow_pickle=True)


# HOW TO LOAD THE DATABASE
"""
b = np.load('a.npy', allow_pickle=True)
"""

if __name__ == "__main__":
    main()
    print("DONE")
