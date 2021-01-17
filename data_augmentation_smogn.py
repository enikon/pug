import numpy as np
import argparse
from data_summary import save_data_summary
import smogn
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--input_train', help="input file for train", default='../dataset/train.npy')
    parser.add_argument('-ot', '--output_train', help="output file for train weights", default='../dataset/train_weights.npy')
    args = parser.parse_args()

    src = np.load(args.input_train, allow_pickle=True)
    src_df = pd.DataFrame(data=src)
    src_df = src_df.add_prefix('col')

    #TODO ERROR custom phi
    smogn_df = smogn.smoter(
        src_df, 'col74',
        k=9,
        rel_coef=0.5, rel_thres=0.02)
    np.save(args.output_eval, smogn_df.to_numpy(), allow_pickle=True)


if __name__ == "__main__":
    main()
    print("DONE")
