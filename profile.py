import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from binascii import hexlify, unhexlify

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert_en', help='convert .pt to .bin if this value is true', default='false')
    parser.add_argument('--size_log_en', help='make size log of raw parameters and compressed parametersif this value is true', default='false')
    parser.add_argument('--src_dir', help='path of directory which contains .pt files', default='dense_weight_pt')
    parser.add_argument('--dst_dir', help='path of directory which save .bin files', default='dense_weight_bin')
    args=parser.parse_args()
    return args

def convert_pt_to_bin(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    file_list = os.listdir(src_dir)
    for i in file_list:
        i_file = src_dir + '/' + i
        o_file = dst_dir + '/' + os.path.splitext(os.path.basename(i))[0] + '.bin'
        #print(o_file)
        pt_file = torch.load(i_file)
        nd_arr = pt_file.detach().cpu().numpy()
        nd_arr.astype(np.float16).tofile(o_file)
    #x = torch.tensor([0.0800, 0.0800, 0.0800, 0.0800, 0.2443, 0.2443, 0.2443, -0.2681], dtype=torch.float16)
    #nd_arr = x.detach().cpu().numpy()
    #print(x)
    #print(nd_arr)
    #nd_arr.astype(np.float16).tofile("tmp.bin")
    #exit()

def make_size_log(args):
    path_dir = args.src_dir
    file_list = os.listdir(path_dir)
    f = open(path_dir+'.log', 'w')
    for i in file_list:
        path_file = path_dir + '/' + i
        if '.gz' not in i:
            print(i, end = ', ', file = f)
            print(os.path.getsize(path_file), end=', ', file = f)
            path_file = path_dir + '/' + i + '.gz'
            print(os.path.getsize(path_file), file = f)
    f.close()


def test():
    pt_file = torch.load("sparse_weight_pt/model.decoder.layers.0.self_attn.q_proj.weight_weights.pt")
    print(pt_file[0][0])
    print(pt_file[0][1])
    nd_arr = pt_file.numpy().reshape(-1)
    df = pd.DataFrame(nd_arr)
    #print(df[0])
    exit()
    nd_arr_unique = np.unique(nd_arr, return_counts=True)
    df = pd.DataFrame(nd_arr_unique)
    df = df.transpose()
    df.to_csv('sample.csv', index = False)
    #print(df)

    print(nd_arr_unique[0])
    exit()
    #print(nd_arr.shape)
    #print(nd_arr.size)
    #print(nd_arr[0].shape)
    #for i in range(nd_arr.shape[0]):
    #    for j in range(nd_arr.shape[1]):
    #        print(nd_arr[i][j])
    plt.hist(nd_arr, bins=100)
    plt.show()
    exit()

    X = list(range(nd_arr.size))
    #X = plt.xlim([0,nd_arr.shape[0]-1])
    #print(X)
    Y = nd_arr
    #Y = [0.8, 0.2, 0.1, 0.01, 0]
    #plt.plot(X, Y, color='red', marker='o', alpha=0.5, linewidth=2)
    plt.plot(X, Y)
    plt.xlim(0,768)
    plt.title("test")
    plt.xlabel("weight")
    plt.ylabel("weight value")
    plt.show()
    #print(pt_file.shape[0]*pt_file.shape[1])
    #print(pt_file.shape[0]*pt_file.shape[1])
    #for i in range(pt_file.shape[0]*pt_file.shape[1]):
    #    print(pt_file[0][i])

    #print(pt_file)
    #print(pt_file[0][0])
    #print(pt_file[0][1])
    #
    #nd_arr = pt_file.detach().cpu().numpy()
    
    #print(pt_file.size())
    #print(nd_arr)
    #print(nd_arr[0][0])
    #print(type(nd_arr[0]))
    #x = torch.float16(1.7517
    
    #nd_arr.astype(np.float16).tofile("bias_data")

def bin_fix(num, k):
    bin_cd = bin(num)
    return bin_cd[2:].zfill(k)

def rle_encode_binary(string, k):
    count_list = []
    listc = [i for i in string]
    list8=[]
    for i in range(len(listc)):
        if i%2 ==0:
            list8.append(listc[i]+listc[i+1])
    print(list8)

    count = 0
    i = 0
    length_max = 0
    previous_character = listc[0]
    while (i <= len(listc) - 1):
        while (listc[i] == previous_character):
            i = i + 1
            count = count + 1
            if i > len(listc) - 1:
                break
        x = [previous_character, count]
        length_max = max(length_max,count)
        count_list.append(x)
        if i > len(listc) - 1:
            break
        previous_character = listc[i]
        count = 0
    max_bits = ((2**k) - 1)
    encode = ""
    for i in range(len(count_list)):
        if count_list[i][1] <= max_bits:
            code = bin_fix(int(count_list[i][0], 16), 4)
            encode = encode + code
            code = bin_fix(count_list[i][1], k)
            encode = encode + code
        else:
            this = count_list[i][1]
            while(this != 0):
                if this > max_bits:
                    code = bin_fix(int(count_list[i][0], 16), 4)
                    encode = encode + code
                    code = bin_fix(max_bits, k)
                    encode = encode + code
                    this = this - max_bits
                else:
                    code = bin_fix(int(count_list[i][0], 16), 4)
                    encode = encode + code
                    code = bin_fix(this, k)
                    encode = encode + code
                    this = 0

    print("Encoded Run Length Sequence is: ", encode)
    print('org len', len(string))
    print('encoded len', len(encode))
    print('length_max', length_max)
    return [encode, k]

def rle():
    with open("tmp.bin","rb") as f:
        data = f.read().hex()
        print(data)
    #data = "11111111111111111111"
    rle_result, k = rle_encode_binary(data, 4)
    bit_strings = [rle_result[i:i+8] for i in range(0, len(rle_result), 8)]
    byte_list = [int(b,2) for b in bit_strings]

    with open('byte.dat', 'wb') as f:
        f.write(bytearray(byte_list))
    exit()

def main(args):
    if args.convert_en == 'true':
        convert_pt_to_bin(args)
    if args.size_log_en == 'true':
        make_size_log(args)
    #test()
    rle()

if __name__ == '__main__':
    args = get_arguments()
    main(args)
