import numpy as np
import argparse

from data_summary import save_data_summary


def main(_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_train', help="input file for train", default='../dataset/train.npy')
    parser.add_argument('-o', '--output_train', help="output file for train",
                        default='../dataset/train_extension.npy')
    parser.add_argument('-cn', '--classes_number', help="how many classes in classification, 0 means regression",
                        default=5, type=int)
    parser.add_argument('-s', '--seed', help="seed for random smote", default=800, type=int)
    parser.add_argument('-ds', '--data_summary', help="data summary file", default='../dataset_summary.txt')

    parser.add_argument('-qh', '--sequence_hours_size', help="length input per one entry", default=24, type=int)
    parser.add_argument('-qd', '--sequence_days_size', help="how many days of the same hour in the input per one entry",
                        default=7, type=int)
    parser.add_argument('-qw', '--sequence_weekdays_size',
                        help="how many days of the same hour and weekday in the input per one cycle", default=7,
                        type=int)
    args = parser.parse_args(*_args)

    if args.classes_number == 0:
        print(
            "There must be more than 1 class, Check -cn parameter while executing data_augmentation_smote.py. Now it is set to " + args.classes_number + ".")
    np.random.seed(args.seed)

    src = np.load(args.input_train, allow_pickle=True)
    x_slice = slice(0, -args.classes_number)
    y_slice = slice(-args.classes_number, None)

    current_val = src[:, y_slice]

    replacing_val = np.zeros(src[0].shape)
    replacing_class = np.zeros(src[0].shape)

    replacing_val[args.sequence_hours_size - 1] = 1.
    replacing_val[args.sequence_hours_size + args.sequence_days_size - 1] = 1.
    replacing_val[args.sequence_hours_size + args.sequence_days_size + args.sequence_weekdays_size - 1] = 1.
    replacing_class[-args.classes_number:] = 1.

    current_stats = np.sum(current_val, axis=0)
    base_max_drop = 0.4
    #normalise_current_val = \
    #    inputs_val + np.outer(
    #        inputs_val[:, args.sequence_hours_size - 1] * np.sum(current_val * (-1 + 1. / (
    #                1. - base_max_theft * np.arange(0, args.classes_number) / (args.classes_number - 1))), axis=1),
    #        replacing_val
    #    )

    corrects = src[np.where(src[:, -args.classes_number] == 1.)]
    alpha = 0.2

    parted = [src]
    for i in range(1, args.classes_number):
        replacing_to_class = np.zeros(src[0].shape)
        replacing_to_class[-args.classes_number+i] = 1.
        missing_size = int(current_stats[0] - current_stats[i])
        corrects_part = \
            corrects * (1 - replacing_val * (base_max_drop * i / (args.classes_number-1)))\
            + (-1) * corrects * replacing_class + alpha/args.classes_number * replacing_class\
            + (1.-alpha) * replacing_to_class
        np.random.shuffle(corrects_part)
        parted.append(corrects_part[:missing_size, :])

    stacked = np.vstack(parted)
    np.random.shuffle(stacked)
    np.save(args.output_train, stacked, allow_pickle=True)

    save_data_summary(args.data_summary, np.sum(np.round(stacked[:, y_slice]), axis=0), 0.0, 1.0 / args.classes_number)


if __name__ == "__main__":
    main({})
    print("DONE")
