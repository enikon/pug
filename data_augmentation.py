import numpy as np
import argparse


def k_bucketing_count_array(
        src, buckets, bucket_pointer, #Arrays
        args, k_buckets_coeff_init
):

    li = src * args.buckets_number
    res = np.zeros((len(src),))
    alpha_coeff = 1.0 / k_buckets_coeff_init

    for ik in range(args.k_bucket_number):
        offset_li = np.clip(li - np.floor(li), 0, 0.5)
        offset_ri = np.clip(np.floor(li) + 1 - li, 0, 0.5)

        res = res +\
            alpha_coeff * (
                offset_li * buckets[bucket_pointer - ik] +
                (0.5-offset_li) * buckets[bucket_pointer - ik - 1]
            ) \
            + alpha_coeff * (
                offset_ri * buckets[bucket_pointer + ik] +
                (0.5 - offset_ri) * buckets[bucket_pointer + ik + 1]
            )
        alpha_coeff *= args.k_bucket_alpha
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--input_train', help="input file for train", default='../dataset/train.npy')
    parser.add_argument('-ie', '--input_eval', help="input file for eval", default='../dataset/eval.npy')
    parser.add_argument('-ot', '--output_train', help="output file for train weights", default='../dataset/train_weights.npy')
    parser.add_argument('-oe', '--output_eval', help="output file for eval weights", default='../dataset/eval_weights.npy')

    parser.add_argument('-b', '--buckets_number', help="number of buckets", default=20, type=int)
    parser.add_argument('-k', '--k_bucket_number', help="number of neighbouring buckets", default=2, type=int)
    parser.add_argument('-a', '--k_bucket_alpha', help="geometric decrease on following buckets", default=1.0/3, type=float)
    args = parser.parse_args()

    src = np.load(args.input_train, allow_pickle=True)

    #BUCKETISATION (FROM TRAIN ONLY)

    point = np.floor(np.minimum(src[:, -1] * args.buckets_number, 19)).astype('int')
    buckets = np.bincount(point)

    ext_buckets = np.zeros(args.k_bucket_number * 2 + len(buckets), dtype='int')
    ext_buckets[args.k_bucket_number:-args.k_bucket_number] = buckets[:]
    ext_point = point + args.k_bucket_number

    #K-COEFF CALCULATION

    alpha_coeff = 1.0
    alpha_max = 0.0

    for ik in range(args.k_bucket_number):
        alpha_max += alpha_coeff
        alpha_coeff *= args.k_bucket_alpha

    res = k_bucketing_count_array(src[:, -1], ext_buckets, ext_point,
                            args, alpha_max)

    max_value = np.max(res)
    np.save(args.output_train, max_value  / res, allow_pickle=True)

    src_eval = np.load(args.input_eval, allow_pickle=True)
    ext_point_eval = np.floor(np.minimum(src_eval[:, -1] * args.buckets_number, 19)).astype('int')\
        + args.k_bucket_number

    res_eval = k_bucketing_count_array(src_eval[:, -1], ext_buckets, ext_point_eval,
                                       args, alpha_max)

    np.save(args.output_eval, max_value / res_eval, allow_pickle=True)


if __name__ == "__main__":
    main()
    print("DONE")
