import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#import cv2

pd.set_option('display.max_rows', 130)
pd.set_option('display.min_rows', 130)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert_en', help='convert .pt to .bin if this value is true', default='false')
    parser.add_argument('--sparsity_en', help='profile sparsity', default='true')
    parser.add_argument('--size_log_en', help='make size log of raw parameters and compressed parametersif this value is true', default='false')
    parser.add_argument('--src_dir', help='path of directory which contains .pt files', default='dense_weight_pt')
    parser.add_argument('--m_size', help='the number of row', default='20480')
    parser.add_argument('--dst_dir', help='path of directory which save .bin files', default='dense_weight_bin')
    args=parser.parse_args()
    return args

def profile_sparsity(args):
    src_dir = args.src_dir
    m_size = args.m_size
    dst_dir = args.dst_dir
    file_list = os.listdir(src_dir)
    file_list=sorted(file_list)
    #img = np.zeros((1024,1), np.uint8)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    #exit()
    for i in file_list:
#        print(i)
        #if i!='activation_out_model.decoder.layers.1.activation_fn_241.pt':
        #    continue
        #if i!='classifier.5.pt':
        #if i!='features.11.pt':
        #    continue
        invalid_pattern_cnt=0
        invalid_pattern_0_zero_cnt=0
        invalid_pattern_1_zero_cnt=0

        i_file = src_dir + '/' + i
        o_file = dst_dir + '/' + os.path.splitext(os.path.basename(i))[0] + '.bin'
        pt_file = torch.load(i_file)
        pt_file = torch.flatten(pt_file)
#        pt_file = torch.tensor([0,0,0,0, 0,1,0,2, 3,4,1,0, 1,1,2,2, 3,4,1,0, 1,1,2,2, 3,0,1,0, 0,0,2,2 ], dtype=torch.float16)
        nd_arr = pt_file.detach().cpu().numpy()
        total_cnt=nd_arr.size
        zero_cnt=np.count_nonzero(nd_arr==0)
        non_zero_cnt = np.count_nonzero(nd_arr)
        zero_ratio = round(zero_cnt/total_cnt*100,2)
        invalid_row =0
        invalid_row_cnt = 0
        invalid_row_indx =[]
        invalid_row_even=0
        invalid_row_cnt_even=0
        invalid_row_odd=0
        invalid_row_cnt_odd=0
#        bitmap = []
#        index =[]
        m_size = int(m_size)
        m_size_split = m_size//2
        for i in range(0, total_cnt, 4):
            if((i%m_size)==0):
                if(invalid_row==1):
                    invalid_row_cnt+=1
                invalid_row =0
            if((i%m_size_split)==0):
                if(invalid_row_even==1):
                    invalid_row_cnt_even+=1
                if(invalid_row_odd==1):
                    invalid_row_cnt_odd+=1
                invalid_row_even =0
                invalid_row_odd =0
            row_index = i//m_size
            #print(nd_arr[i:i+4])
            #print(np.count_nonzero(nd_arr[i:i+4]==0))
            if (np.count_nonzero(nd_arr[i:i+4]==0))<2:
#                index.append(i)
                invalid_pattern_cnt+=1
                invalid_row=1
                if(((i%m_size)//m_size_split)==0):
                  invalid_row_even=1
                else:
                  invalid_row_odd=1
                  
                
            if (np.count_nonzero(nd_arr[i:i+4]==0))==0:
                invalid_pattern_0_zero_cnt+=1
            elif (np.count_nonzero(nd_arr[i:i+4]==0))==1:
                invalid_pattern_1_zero_cnt+=1
        if(invalid_row==1):
            invalid_row_cnt+=1
        if(invalid_row_even==1):
            invalid_row_cnt_even+=1
        if(invalid_row_odd==1):
            invalid_row_cnt_odd+=1
        invalid_ratio = round(invalid_pattern_cnt/(total_cnt/4)*100,2)
        print(i_file, total_cnt, non_zero_cnt, zero_cnt, zero_ratio,'%', invalid_pattern_cnt, total_cnt/4, invalid_ratio,'%', invalid_pattern_0_zero_cnt, invalid_pattern_1_zero_cnt, invalid_row_cnt, invalid_row_cnt_even, invalid_row_cnt_odd)
#        print(index)
        #print(bitmap)
        #print(nd_arr)
#        exit()
        #nd_arr.astype(np.float32).tofile(o_file)

def convert_pt_to_bin(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    file_list = os.listdir(src_dir)
    for i in file_list:
        m2n_cnt=0
        i_file = src_dir + '/' + i
        o_file = dst_dir + '/' + os.path.splitext(os.path.basename(i))[0] + '.bin'
        #print(o_file)
        pt_file = torch.load(i_file)
        #print(pt_file.dtype)
        #print(pt_file.shape)
        pt_file = torch.flatten(pt_file)
        #pt_file = pt_file[0:12]
        #total_cnt = pt_file.shape[0]
        #non_zero_cnt = torch.count_nonzero(pt_file)
        #zero_cnt = total_cnt - non_zero_cnt
        #zero_ratio = zero_cnt/total_cnt*100
        #print(total_cnt, non_zero_cnt, zero_cnt, zero_ratio)
        #print(pt_file.dtype)
        #print(pt_file.shape)
        nd_arr = pt_file.detach().cpu().numpy()
        total_cnt=nd_arr.size
        zero_cnt=np.count_nonzero(nd_arr==0)
        non_zero_cnt = np.count_nonzero(nd_arr)
        zero_ratio = round(zero_cnt/total_cnt*100,2)
        for i in range(0, total_cnt, 4):
            #print(nd_arr[i:i+4])
            #print(np.count_nonzero(nd_arr[i:i+4]==0))
            if (np.count_nonzero(nd_arr[i:i+4]==0))<2:
                m2n_cnt+=1
        invalid_ratio = round(m2n_cnt/(total_cnt/4)*100,2)
        print(i_file, total_cnt, non_zero_cnt, zero_cnt, zero_ratio,'%', m2n_cnt,'/', total_cnt/4, invalid_ratio,'%')
        #print(nd_arr)
        #exit()
        nd_arr.astype(np.float32).tofile(o_file)
    #x = torch.tensor([0.0800, 0.0800, 0.0800, 0.0800, 0.0000, 0.0000, 0.0000, 0.0000, 0.2443, 0.2443, 0.2443, 0.0000, 0.0000, 0.0000,  -0.2681, -0.2681], dtype=torch.float16)
    #x = torch.tensor([0.0800, 0.0800, 0.0800, 0.0800, 0.2443, 0.2443, 0.2443, -0.2681], dtype=torch.float16)
    #nd_arr = x.detach().cpu().numpy()
    #print(x)
    #print(nd_arr)
    #nd_arr.astype(np.float16).tofile("tmp2.bin")
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


def test(input_file):
    with open(input_file,"rb") as f:
        data = f.read().hex()
    print(data)
    exit()
    #data = bytes.fromhex(data)
    #nd_arr = np.frombuffer(data, dtype= np.float16)
    #print(nd_arr)

    list8=[]
    list16=[]
    list_exp=[]
    for i in range(len(data)):
        if i%2 ==0:
            list8.append(data[i]+data[i+1])
    for i in range(len(list8)):
        if i%2 ==0:
            list16.append(list8[i]+list8[i+1])
    for i in list16:
        #print(int(extract_exp(i),2))
        list_exp.append(int(extract_exp(i),2))
    #print(list_exp)

########
    k=4
    count_list = []
    lists = list_exp

    count = 0
    i = 0
    length_max = 0
    length_acc = 0
    run_cnt = 0
    previous_character = lists[0]
    while (i <= len(lists) - 1):
        while (lists[i] == previous_character):
            i = i + 1
            count = count + 1
            if i > len(lists) - 1:
                break
        x = [previous_character, count]
        length_max = max(length_max,count)
        length_acc += count
        count_list.append(x)
        if i > len(lists) - 1:
            break
        previous_character = lists[i]
        count = 0
    max_bits = ((2**k) - 1)
    encode = ""
    for i in range(len(count_list)):
        if count_list[i][1] <= max_bits:
            code = bin_fix(count_list[i][0], 16)
            encode = encode + code
            code = bin_fix(count_list[i][1], k)
            encode = encode + code
            run_cnt +=1
        else:
            this = count_list[i][1]
            while(this != 0):
                if this > max_bits:
                    code = bin_fix(count_list[i][0], 16)
                    encode = encode + code
                    code = bin_fix(max_bits, k)
                    encode = encode + code
                    this = this - max_bits
                    run_cnt +=1
                else:
                    code = bin_fix(count_list[i][0], 16)
                    encode = encode + code
                    code = bin_fix(this, k)
                    encode = encode + code
                    this = 0
                    run_cnt +=1

    #print("Encoded Run Length Sequence is: ", encode)
    #print('org len', len(string))
    #print('encoded len', len(encode))
    print('length_max', length_max)
    #print('length_acc', length_acc)
    #print('run_cnt', run_cnt)
    print('length_avg: {0:.2f}'.format(length_acc/run_cnt))
    exit()


##########
    nd_arr = np.array(list_exp)
    #exit()


    #input_file = input_file.replace('bin', 'pt')
    #pt_file = torch.load(input_file)
    #nd_arr = pt_file.numpy().reshape(-1)
    
    # data value count
    df = pd.DataFrame(nd_arr)
    print(df.value_counts())
    exit()

    # unique value count
    #nd_arr_unique = np.unique(nd_arr, return_counts=True)
    #df = pd.DataFrame(nd_arr_unique)
    #pd.set_option('display.max_row', 20)
    #print(df.value_counts())
    #df = df.transpose()
    #df.to_csv('sample.csv', index = False)
    #print("# of unique value: ", df.count()[0])
    #print("# of total       : ", df.sum(axis='rows')[1])
    #print(df)

    #print(nd_arr_unique[0])
    #print(nd_arr.shape)
    #print(nd_arr.size)
    #print(nd_arr[0].shape)
    #for i in range(nd_arr.shape[0]):
    #    for j in range(nd_arr.shape[1]):
    #        print(nd_arr[i][j])

    #plt.hist(nd_arr, bins=100)
    #plt.show()
    #exit()

    
    # plot histogram
    X = list(range(nd_arr.size))
    Y = nd_arr
    #plt.plot(X, Y, color='red', marker='o', alpha=0.5, linewidth=2)
    plt.plot(X, Y)
    plt.xlim(0,768*2)
    plt.title("test")
    plt.xlabel("weight")
    plt.ylabel("weight value")
    plt.show()

def bin_fix(num, k):
    bin_cd = bin(num)
    return bin_cd[2:].zfill(k)

def bin_to_byte(data):
    bit_strings = [data[i:i+8] for i in range(0, len(data), 8)]
    byte_list = [int(b,2) for b in bit_strings if len(b) == 8 ]
    if len(bit_strings[-1])!=8:
        shift_bit = 8 - len(bit_strings[-1])
        byte_list.append(int(bit_strings[-1],2)<<shift_bit)
    return byte_list

def extract_exp(data):
    code = bin(int(data, 16))[2:].zfill(16)
    return code[8]
    #return code[0:8]+code[14:16]
    #return code[9:14]
    #return code[8:14]
    #return code[0:16]

def m2n_encode_binary(string, m, n):
    list16=[]
    for i in range(len(string)):
        if i%4 == 0:
            list16.append(string[i:i+4])
    zero_avg = 0
    zero_count = 0
    zero_dist = []
    for i in range(len(list16)):
        if i%m == 0:
            zero_count = 0
            for j in range(m):
                if list16[i+j] == "0000":
                    zero_count += 1
            zero_dist.append(zero_count)
            zero_avg += zero_count

    df = pd.DataFrame(zero_dist)
    print(df.value_counts())
    #print(math.ceil(math.log2(9)))
    #print(zero_avg)
    #print(len(list16)/m)
    zero_avg /= len(list16)/m
    print(zero_avg)
    #print(zero_dist)
    #print(df)
    #exit()

    


def rle_encode_binary(string, k):
    count_list = []
    listc = [i for i in string]
    list8=[]
    list16=[]
    for i in range(len(listc)):
        if i%2 ==0:
            list8.append(listc[i]+listc[i+1])
    for i in range(len(list8)):
        if i%2 ==0:
            list16.append(list8[i]+list8[i+1])
    #print(list8)
    #print(list16)

    lists = list16

    count = 0
    i = 0
    length_max = 0
    length_acc = 0
    run_cnt = 0
    previous_character = lists[0]
    zero_dist = []
    while (i <= len(lists) - 1):
        while (lists[i] == previous_character):
            i = i + 1
            count = count + 1
            if i > len(lists) - 1:
                break
        x = [previous_character, count]
        length_max = max(length_max,count)
        if x[0]=='0000':
            zero_dist.append(x[1])
        length_acc += count
        count_list.append(x)
        if i > len(lists) - 1:
            break
        previous_character = lists[i]
        count = 0
    df = pd.DataFrame(zero_dist)
    print(df)
    max_bits = ((2**k) - 1)
    encode = ""
    for i in range(len(count_list)):
        if count_list[i][1] <= max_bits:
            code = bin_fix(int(count_list[i][0], 16), 16)
            encode = encode + code
            code = bin_fix(count_list[i][1], k)
            encode = encode + code
            run_cnt +=1
        else:
            this = count_list[i][1]
            while(this != 0):
                if this > max_bits:
                    code = bin_fix(int(count_list[i][0], 16), 16)
                    encode = encode + code
                    code = bin_fix(max_bits, k)
                    encode = encode + code
                    this = this - max_bits
                    run_cnt +=1
                else:
                    code = bin_fix(int(count_list[i][0], 16), 16)
                    encode = encode + code
                    code = bin_fix(this, k)
                    encode = encode + code
                    this = 0
                    run_cnt +=1

    #print("Encoded Run Length Sequence is: ", encode)
    #print('org len', len(string))
    #print('encoded len', len(encode))
    print('length_max', length_max)
    #print('length_acc', length_acc)
    #print('run_cnt', run_cnt)
    print('length_avg: {0:.2f}'.format(length_acc/run_cnt))
    return [encode, k]

def rle_decode_binary(encoded, k):
    #print(encoded)
    #print(k)
    encoded_bin = ""
    for i in encoded:
        encoded_bin = encoded_bin + bin_fix(int(i,16),4)
    #print(encoded_bin)

    seq_mas = []
    while encoded_bin:
        if len(encoded_bin) >= 16+k:
            seq_mas.append(encoded_bin[:16+k])
            encoded_bin = encoded_bin[16+k:]
        else:
            break
    #print(seq_mas)
    decode = ""
    int_val = []
    max_bits = ((2 ** k) - 1)
    for i in range(len(seq_mas)):
        length = int(seq_mas[i][16:17+k],2)
        run = seq_mas[i][:16]*length
        decode = decode + run
    return decode

def rle_profile(input_file, length_bit):
    with open(input_file,"rb") as f:
        data = f.read().hex()
    rle_encode_result, k = rle_encode_binary(data, length_bit)
    byte_list = bin_to_byte(rle_encode_result)

    output_file = input_file + ".rle"

    with open(output_file, 'wb') as f:
        f.write(bytearray(byte_list))
    print("Raw size         :   {0} Byte".format(os.path.getsize(input_file)))
    print("RLE encoded size :   {0} Byte".format(os.path.getsize(output_file)))
    print("Compression ratio:   {0:.2f} %".format((os.path.getsize(input_file)-os.path.getsize(output_file))/os.path.getsize(input_file)*100))

    #with open(output_file, "rb") as f:
    #    data = f.read().hex()
    #rle_decode_result = rle_decode_binary(data, length_bit)
    #byte_list = bin_to_byte(rle_decode_result)
    #with open('decode.dat', 'wb') as f:
    #    f.write(bytearray(byte_list))

def m2n_profile(input_file, m, n):
    with open(input_file,"rb") as f:
        data = f.read().hex()
    m2n_encode_result = m2n_encode_binary(data, m, n)

def main(args):
    if args.sparsity_en == 'true':
        profile_sparsity(args)
    if args.convert_en == 'true':
        convert_pt_to_bin(args)
    exit()
    if args.size_log_en == 'true':
        make_size_log(args)

    #input_file = "tmp2.bin"
    input_file = "sparse_weight_bin/model.decoder.layers.0.self_attn.q_proj.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.fc1.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.fc2.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.self_attn.k_proj.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.self_attn.out_proj.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.self_attn.q_proj.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.self_attn.v_proj.weight_weights.bin"
    #input_file = "sparse_weight_bin/model.decoder.layers.0.final_layer_norm.weight_weights.bin"

    ################################################################################################
    ## RLE
    ################################################################################################
    #rle_profile(input_file, 2)

    ################################################################################################
    ## M to N 
    ################################################################################################
    m=4
    print(m)
    m2n_profile(input_file, m, m/2)
    m=8
    print(m)
    m2n_profile(input_file, m, m/2)
    m=16
    print(m)
    m2n_profile(input_file, m, m/2)
    m=32
    print(m)
    m2n_profile(input_file, m, m/2)
    m=64
    print(m)
    m2n_profile(input_file, m, m/2)
    m=128
    print(m)
    m2n_profile(input_file, m, m/2)

    #test(input_file)

if __name__ == '__main__':
    args = get_arguments()
    main(args)
