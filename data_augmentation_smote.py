import numpy as np
import argparse
from imblearn.over_sampling import SMOTE

from data_summary import save_data_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--input_train', help="input file for train", default='../dataset_classes/train.npy')
    parser.add_argument('-ot', '--output_train', help="output file for train", default='../dataset_classes/train_smote.npy')
    parser.add_argument('-cn', '--classes_number', help="how many classes in classification, 0 means regression",
                        default=5, type=int)
    parser.add_argument('-s', '--seed', help="seed for random smote", default=800, type=int)
    parser.add_argument('-ds', '--data_summary', help="data summary file", default='../dataset_summary.txt')

    args = parser.parse_args()

    if args.classes_number == 0:
        print("There must be more than 1 class, Check -cn parameter while executing data_augmentation_smote.py. Now it is set to "+args.classes_number+".")

    src = np.load(args.input_train, allow_pickle=True)
    oversample = SMOTE()

    x_slice = slice(0, -args.classes_number)
    y_slice = slice(-args.classes_number, None)

    dst, y = oversample.fit_resample(src[:, x_slice], src[:, y_slice])
    resampled = np.concatenate((dst, y), axis=1)

    np.random.seed(args.seed)
    np.random.shuffle(resampled)

    np.save(args.output_train, resampled, allow_pickle=True)

    save_data_summary(args.data_summary, np.sum(resampled[:, y_slice], axis=0), 0.0, 1.0 / args.classes_number)


if __name__ == "__main__":
    main()
    print("DONE")
