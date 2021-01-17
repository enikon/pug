import argparse
import os
import data_augmentation_smote_under
import data_augmentation_extension


def main(_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input folder", default='../dataset')
    parser.add_argument('-e', '--extension', help="file extension", default='_base')
    args = parser.parse_args(*_args)

    # data_augmentation_extension.main(
    #     ([
    #         "-i", os.path.join(args.input, 'train.npy'),
    #         "-o", os.path.join(args.input, 'train' + args.extension + '.npy'),
    #         "-cn", '4'
    #     ],)
    # )
    data_augmentation_smote_under.main(
        ([
             "-i", os.path.join(args.input, 'train.npy'),
             "-o", os.path.join(args.input, 'train' + args.extension + '.npy'),
             "-cn", '4',
             "-tc", '100000'
         ],)
    )
    data_augmentation_smote_under.main(
        ([
            "-i", os.path.join(args.input, 'eval.npy'),
            "-o", os.path.join(args.input, 'eval' + args.extension + '.npy'),
            "-cn", '4',
            "-tc", '40000'
        ],)
    )
    data_augmentation_smote_under.main(
        ([
            "-i", os.path.join(args.input, 'test.npy'),
            "-o", os.path.join(args.input, 'test' + args.extension + '.npy'),
            "-cn", '4',
            "-tc", '-1'
        ],)
    )


if __name__ == "__main__":
    main({})
    print("DONE")
