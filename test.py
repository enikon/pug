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

def main():
    BATCH_SIZE = 64
    EPOCHS = 1
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

    test_set = np.load(os.path.join(args.input, 'test.npy'), allow_pickle=True)

    test_predict = test_set[:, -1]

    show_confusion_matrix(
        np.expand_dims(np.expand_dims(test_set[:, -1], -1), -1),
        test_predict
    )


if __name__ == "__main__":
    main()
