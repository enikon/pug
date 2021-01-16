import numpy as np
import argparse
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


def main(_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input file", default='../dataset_classes/train.npy')
    parser.add_argument('-o', '--output', help="output file", default='../dataset_classes/train_smote_under.npy')
    parser.add_argument('-cn', '--classes_number', help="how many classes in classification, 0 means regression",
                        default=5, type=int)
    parser.add_argument('-tc', '--target_count', help="how many entries will there be for each class", default=int(1e6),
                        type=int)
    parser.add_argument('-s', '--seed', help="seed for random smote", default=800, type=int)
    args = parser.parse_args(*_args)

    if args.classes_number == 0:
        print("There must be more than 1 class, Check -cn parameter while executing data_augmentation_smote.py. Now it is set to "+args.classes_number+".")

    src = np.load(args.input, allow_pickle=True)

    x_slice = slice(0, -args.classes_number)
    y_slice = slice(-args.classes_number, None)

    current_val = np.sum(src[:, y_slice], axis=0)
    steps = []

    target_val = args.target_count
    lower_val = np.min(current_val)
    higher_val = np.max(current_val)

    if target_val > lower_val:
        smote_target_val = min(target_val, higher_val)
        over_val = np.maximum(current_val, smote_target_val).astype(int)
        oversample = SMOTE(sampling_strategy=dict(enumerate(over_val)))
        steps.append(('o', oversample))
    else:
        over_val = current_val

    if target_val < higher_val:
        smote_target_val = max(target_val, lower_val)
        under_val = np.minimum(over_val, smote_target_val).astype(int)
        undersample = RandomUnderSampler(sampling_strategy=dict(enumerate(under_val)))
        steps.append(('u', undersample))

    pipeline = Pipeline(steps=steps)

    dst, y = pipeline.fit_resample(src[:, x_slice], src[:, y_slice])
    resampled = np.concatenate((dst, y), axis=1)

    np.random.seed(args.seed)
    np.random.shuffle(resampled)

    np.save(args.output, resampled, allow_pickle=True)


if __name__ == "__main__":

    main({})
    print("DONE")
