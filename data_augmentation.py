import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input root folder", default='../dataset/train.npy')
    parser.add_argument('-o', '--output', help="output root folder", default='../dataset/train_weights.npy')
    parser.add_argument('-b', '--buckets_number', help="number of buckets", default=20, type=int)
    parser.add_argument('-k', '--k_bucket_number', help="number of neighbouring buckets", default=2, type=int)
    parser.add_argument('-a', '--k_bucket_alpha', help="geometric decrease on following buckets", default=1.0/3, type=float)
    args = parser.parse_args()

    src = np.load(args.input, allow_pickle=True)
    point = np.floor(np.minimum(src[:, -1] * args.buckets_number, 19)).astype('int')
    buckets = np.bincount(point)

    alpha_coeff = 1.0
    alpha_max = 0.0

    for ik in range(args.k_bucket_number):
        alpha_max += alpha_coeff
        alpha_coeff *= args.k_bucket_alpha

    alpha_coeff = 1.0 / alpha_max

    ext_buckets = np.zeros(args.k_bucket_number*2 + len(buckets), dtype='int')
    ext_buckets[args.k_bucket_number:-args.k_bucket_number] = buckets[:]
    ext_point = point  + args.k_bucket_number

    li = src[:, -1] * args.buckets_number
    res = np.zeros((len(src),))

    for ik in range(args.k_bucket_number):
        offset_li = np.clip(li - np.floor(li), 0, 0.5)
        offset_ri = np.clip(np.floor(li) + 1 - li, 0, 0.5)

        res = res +\
            alpha_coeff * (
                offset_li * ext_buckets[ext_point - ik] +
                (0.5-offset_li) * ext_buckets[ext_point - ik - 1]
            ) \
            + alpha_coeff * (
                offset_ri * ext_buckets[ext_point + ik] +
                (0.5 - offset_ri) * ext_buckets[ext_point + ik + 1]
            )
        alpha_coeff *= args.k_bucket_alpha

    max_value = np.max(res)
    np.save(args.output, max_value  / res / 10, allow_pickle=True)


if __name__ == "__main__":
    main()
    print("DONE")
