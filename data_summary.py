def save_data_summary(path, src, brackets_start, brackets_size):
    with open(path, 'w') as file:
        b = brackets_start
        for i in src:
            file.write("{:.3f}:{:.3f}\t{:d}\n".format(b, b+brackets_size, int(i)))
            b += brackets_size
