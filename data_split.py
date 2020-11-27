import pandas as pd
import os
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input root folder", default='../raw_dataset')
    parser.add_argument('-o', '--output', help="output root folder", default='../dataset')
    parser.add_argument('-se', '--split_eval', help="percentage of data to be in evaluation set", default=0.1, type=float)
    parser.add_argument('-st', '--split_test', help="percentage of data to be in test set", default=0.1, type=float)

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print("MyError: No input directory found.")
        return

    # Clear directory
    # TODO DONE WARNING failsafe
    if os.path.realpath(args.output) == "D:\\Mikolaj\\Desktop\\PUG\\dataset" \
            and os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    for input_dirs in os.listdir(args.input):
        cat_dir = os.path.join(args.output, input_dirs)
        os.makedirs(cat_dir)
        input_cat_dir = os.path.join(args.input, input_dirs)
        for input_file in os.listdir(input_cat_dir):
            df = pd.read_csv(
                filepath_or_buffer=os.path.join(input_cat_dir, input_file),
                sep=",",
                decimal="."
            )

            # TODO MAIN CONVERT FORMAT
            # TODO MAIN TRAIN/TEST/VAL SPLIT
            # TODO DISCUSS  TRAIN/TEST/VAL SPLIT
            #   a) last x months ? (val vs test)
            #   b) x percentage of users ?

            df.to_csv(
                os.path.join(cat_dir, os.path.basename(input_file)),
                header=None,
                index=False,
                float_format='%.3f'
            )


if __name__ == "__main__":
    main()