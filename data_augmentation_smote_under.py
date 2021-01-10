from collections import Counter

import numpy as np
import argparse
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--input_train', help="input file for train", default='../dataset_classes/train.npy')
    parser.add_argument('-ot', '--output_train', help="output file for train", default='../dataset_classes/train_smote_under.npy')
    parser.add_argument('-cn', '--classes_number', help="how many classes in classification, 0 means regression",
                        default=5, type=int)
    parser.add_argument('-s', '--seed', help="seed for random smote", default=800, type=int)
    args = parser.parse_args()

    if args.classes_number == 0:
        print("There must be more than 1 class, Check -cn parameter while executing data_augmentation_smote.py. Now it is set to "+args.classes_number+".")

    src = np.load(args.input_train, allow_pickle=True)

    val=1e5
    oversample = SMOTE(sampling_strategy={0: 1e5, 1:1e5, 2:1e5, 3:1e5, 4:1e5})
    undersample = RandomUnderSampler(sampling_strategy={0:1e5, 1:1e5, 2:1e5, 3:1e5, 4:1e5})
    steps = [('o', oversample), ('u', undersample)]
    pipeline = Pipeline(steps=steps)

    x_slice = slice(0, -args.classes_number)
    y_slice = slice(-args.classes_number, None)

    dst, y = pipeline.fit_resample(src[:, x_slice], src[:, y_slice])
    resampled = np.concatenate((dst, y), axis=1)

    np.random.seed(args.seed)
    np.random.shuffle(resampled)

    np.save(args.output_train, resampled, allow_pickle=True)


if __name__ == "__main__":
    main()
    print("DONE")
