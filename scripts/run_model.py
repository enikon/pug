import argparse
import os
import mk_model


def main(_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input folder", default='../dataset')
    parser.add_argument('-e', '--extension', help="file extension", default='base')
    parser.add_argument('-m', '--model', help="model name", default='new_baseline')
    args = parser.parse_args(*_args)

    mk_model.main(
        ([
             "-i", os.path.join(args.input),
             "-cn", '4',
             "-iw", "False",
             "-m", args.model,
             "-tt", "train" + args.extension,
             "-et", "eval" + args.extension,
             "-st", "test" + args.extension,
             #"-qh", '48',
             #"-qd", '14',
             #"-qw", '10'
         ],)
    )


if __name__ == "__main__":
    main({})
    print("DONE")
