import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', help='path of directory which contains .pt files', default='dense_pt')
    args=parser.parse_args()
    return args

def main(args):
    path_dir = args.path_dir
    file_list = os.listdir(path_dir)
    f = open(path_dir+'.log', 'w')
    for i in file_list:
        path_file = path_dir + '/' + i
        if '.gz' not in i:
            print(i, end = ' ', file = f)
            print(os.path.getsize(path_file), end=' ', file = f)
            path_file = path_dir + '/' + i + '.gz'
            print(os.path.getsize(path_file), file = f)
    f.close()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
