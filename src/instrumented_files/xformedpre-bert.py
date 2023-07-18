import sys
from instrument_lib import *
import numpy as np
import math


def balance_random_3d(depth, l, wid):
    print('enter scope 1')
    print(1, 4)
    depth_1 = depth
    l_1 = l
    wid_1 = wid
    print(3, 5)
    arr_1 = np.random.rand(depth_1, l_1, wid_1)
    print(3, 6)
    neg_1 = True
    for i_1 in range(depth_1):
        for j_1 in range(l_1):
            for k_1 in range(wid_1):
                if neg_1:
                    print(11, 11)
                    arr_1[i_1][j_1][k_1] *= -1
                print(12, 12)
                neg_1 = not neg_1
    print('exit scope 1')
    return arr_1
    print('exit scope 1')


def balance_random_2d(l, wid):
    print('enter scope 2')
    print(1, 15)
    l_2 = l
    wid_2 = wid
    print(16, 16)
    arr_2 = np.random.rand(l_2, wid_2)
    print(16, 17)
    neg_2 = True
    for i_2 in range(l_2):
        for j_2 in range(wid_2):
            if neg_2:
                print(22, 21)
                arr_2[i_2][j_2] *= -1
            print(23, 22)
            neg_2 = not neg_2
    print('exit scope 2')
    return arr_2
    print('exit scope 2')


def arr_add(dst, src):
    print('enter scope 3')
    print(1, 25)
    dst_3 = dst
    src_3 = src
    for i_3 in range(len(dst_3)):
        for j_3 in range(len(dst_3[0])):
            print(30, 28)
            dst_3[i_3][j_3] += src_3[i_3][j_3]
    print('exit scope 3')
    return dst_3
    print('exit scope 3')


def reLU(img):
    print('enter scope 4')
    print(1, 31)
    img_4 = img
    for i_4 in range(len(img_4)):
        for j_4 in range(len(img_4[0])):
            print(38, 34)
            img_4[i_4][j_4] = max(img_4[i_4][j_4], 0)
    print('exit scope 4')
    return img_4
    print('exit scope 4')


def get_mean(row):
    print('enter scope 5')
    print(1, 37)
    row_5 = row
    print(43, 38)
    sum_val_5 = 0
    for i_5 in range(len(row_5)):
        print(45, 40)
        sum_val_5 += row_5[i_5]
    print('exit scope 5')
    return sum_val_5 / len(row_5)
    print('exit scope 5')


def std_dev(row):
    print('enter scope 6')
    print(1, 43)
    row_6 = row
    print(50, 44)
    result_6 = 0
    for i_6 in range(len(row_6)):
        print(52, 46)
        diff_6 = row_6[i_6] - get_mean(row_6)
        print(52, 47)
        result_6 += diff_6 * diff_6
    print('exit scope 6')
    return math.sqrt(result_6 / len(row_6))
    print('exit scope 6')


def BN_layer(arr, weights, biases):
    print('enter scope 7')
    print(1, 50)
    arr_7 = arr
    weights_7 = weights
    biases_7 = biases
    for i_7 in range(len(arr_7)):
        print(58, 52)
        dev_7 = std_dev(arr_7[i_7])
        print(58, 53)
        mean_7 = get_mean(arr_7[i_7])
        if dev_7 == 0:
            print(60, 54)
            dev_7 = 1
        for j_7 in range(len(arr_7[0])):
            print(62, 56)
            arr_7[i_7][j_7] = weights_7[i_7] * ((arr_7[i_7][j_7] - mean_7) /
                dev_7) + biases_7[i_7]
    print('exit scope 7')
    return arr_7
    print('exit scope 7')


def fc_layer(arr, W, W_0):
    print('enter scope 8')
    print(1, 59)
    arr_8 = arr
    W_8 = W
    W_0_8 = W_0
    print(67, 60)
    result_8 = np.zeros(len(W_8[0]))
    for i_8 in range(len(W_8[0])):
        print(69, 62)
        sum_val_8 = W_0_8[i_8]
        for j_8 in range(len(arr_8)):
            print(72, 64)
            sum_val_8 += arr_8[j_8] * W_8[j_8][i_8]
        print(73, 65)
        result_8[i_8] = sum_val_8
    print('exit scope 8')
    return result_8
    print('exit scope 8')


def softmax(arr):
    print('enter scope 9')
    print(1, 68)
    arr_9 = arr
    print(77, 69)
    sum_val_9 = 0
    for i_9 in range(len(arr_9)):
        print(79, 70)
        sum_val_9 += math.exp(arr_9[i_9])
    print(80, 71)
    result_9 = np.zeros(len(arr_9))
    for i_9 in range(len(arr_9)):
        print(82, 72)
        result_9[i_9] = math.exp(arr_9[i_9]) / sum_val_9
    print('exit scope 9')
    return result_9
    print('exit scope 9')


def concat(emb, head, tokens, d_k, cur):
    print('enter scope 10')
    print(1, 75)
    emb_10 = emb
    head_10 = head
    tokens_10 = tokens
    d_k_10 = d_k
    cur_10 = cur
    for i_10 in range(tokens_10):
        for j_10 in range(d_k_10):
            print(90, 78)
            emb_10[i_10][j_10 + head_10 * d_k_10] = cur_10[i_10][j_10]
    print('exit scope 10')
    return emb_10
    print('exit scope 10')


def self_attn(head, tokens, d_k, Q, K, V):
    print('enter scope 11')
    print(1, 81)
    head_11 = head
    tokens_11 = tokens
    d_k_11 = d_k
    Q_11 = Q
    K_11 = K
    V_11 = V
    print(95, 82)
    scores_11 = np.zeros((tokens_11, tokens_11))
    for i_11 in range(tokens_11):
        for j_11 in range(tokens_11):
            for k_11 in range(d_k_11):
                print(101, 86)
                scores_11[i_11][j_11] += Q_11[head_11][i_11][k_11] * K_11[
                    head_11][j_11][k_11]
    for i_11 in range(tokens_11):
        for j_11 in range(tokens_11):
            print(105, 89)
            scores_11[i_11][j_11] /= math.sqrt(d_k_11)
        print(106, 91)
        scores_11 = np.random.rand(tokens_11, tokens_11)
        print(106, 92)
        scores_11[i_11] = softmax(scores_11[i_11])
    print(104, 93)
    out_11 = np.zeros((tokens_11, d_k_11))
    for i_11 in range(tokens_11):
        for j_11 in range(d_k_11):
            for k_11 in range(tokens_11):
                print(112, 97)
                out_11[i_11][j_11] += scores_11[i_11][k_11] * V_11[head_11][
                    k_11][j_11]
    print('exit scope 11')
    return out_11
    print('exit scope 11')


def main():
    print('enter scope 12')
    print(1, 101)
    print(117, 102)
    d_model_12, heads_12, tokens_12, layers_12 = 12, 12, 8, 12
    print(117, 103)
    d_k_12 = d_model_12 // heads_12
    print(117, 104)
    embeddings_12 = np.random.rand(tokens_12, d_model_12)
    for i_12 in range(tokens_12):
        for j_12 in range(d_model_12):
            if j_12 % 2 == 0:
                print(123, 108)
                embeddings_12[i_12][j_12] += math.sin(i_12 / math.pow(10000,
                    2 * j_12 / d_model_12))
            else:
                print(125, 110)
                embeddings_12[i_12][j_12] += math.cos(i_12 / math.pow(10000,
                    2 * j_12 / d_model_12))
    print(120, 111)
    W_Q_12 = balance_random_3d(heads_12, d_model_12, d_k_12)
    print(120, 112)
    W_K_12 = balance_random_3d(heads_12, d_model_12, d_k_12)
    print(120, 113)
    W_V_12 = balance_random_3d(heads_12, d_model_12, d_k_12)
    print(120, 114)
    Q_12 = np.zeros((heads_12, tokens_12, d_k_12))
    print(120, 115)
    K_12 = np.zeros((heads_12, tokens_12, d_k_12))
    print(120, 116)
    V_12 = np.zeros((heads_12, tokens_12, d_k_12))
    for i_12 in range(heads_12):
        for j_12 in range(tokens_12):
            for k_12 in range(d_k_12):
                print(131, 120)
                sumQ_12, sumK_12, sumV_12 = 0, 0, 0
                for a_12 in range(d_model_12):
                    print(134, 122)
                    sumQ_12 += embeddings_12[j_12][a_12] * W_Q_12[i_12][a_12][
                        k_12]
                    print(134, 123)
                    sumK_12 += embeddings_12[j_12][a_12] * W_K_12[i_12][a_12][
                        k_12]
                    print(134, 124)
                    sumV_12 += embeddings_12[j_12][a_12] * W_V_12[i_12][a_12][
                        k_12]
                print(135, 125)
                Q_12[i_12][j_12][k_12] = sumQ_12
                print(135, 126)
                K_12[i_12][j_12][k_12] = sumK_12
                print(135, 127)
                V_12[i_12][j_12][k_12] = sumV_12
    for i_12 in range(layers_12):
        print(136, 129)
        emb_cpy_12 = np.copy(embeddings_12)
        print(136, 130)
        multi_head_out_12 = np.zeros((tokens_12, d_model_12))
        for j_12 in range(heads_12):
            print(139, 132)
            cur_12 = self_attn(j_12, tokens_12, d_k_12, Q_12, K_12, V_12)
            print(139, 133)
            multi_head_out_12 = concat(multi_head_out_12, j_12, tokens_12,
                d_k_12, cur_12)
        print(140, 134)
        W_attn_12 = np.random.rand(d_model_12, d_model_12)
        for i_12 in range(tokens_12):
            for j_12 in range(d_model_12):
                print(144, 137)
                sum_val_12 = 0
                for k_12 in range(d_model_12):
                    print(147, 139)
                    sum_val_12 += multi_head_out_12[i_12][k_12] * W_attn_12[
                        k_12][j_12]
                print(148, 140)
                embeddings_12[i_12][j_12] = sum_val_12
        print(143, 141)
        embeddings_12 = arr_add(embeddings_12, emb_cpy_12)
        print(143, 142)
        weights_12, biases_12 = np.random.rand(d_model_12), np.random.rand(
            d_model_12)
        print(143, 143)
        embeddings_12 = BN_layer(embeddings_12, weights_12, biases_12)
        print(143, 144)
        emb_cpy_12 = np.copy(embeddings_12)
        print(143, 145)
        W_12 = np.random.rand(d_model_12, d_model_12 * 4)
        print(143, 146)
        W_0_12 = np.random.rand(d_model_12 * 4)
        print(143, 147)
        emb_new_12 = np.zeros((tokens_12, d_model_12 * 4))
        for i_12 in range(tokens_12):
            print(150, 149)
            emb_new_12[i_12] = fc_layer(embeddings_12[i_12], W_12, W_0_12)
        print(151, 150)
        embeddings_12 = emb_new_12
        print(151, 151)
        embeddings_12 = reLU(embeddings_12)
        print(151, 152)
        W_12 = np.random.rand(d_model_12 * 4, d_model_12)
        print(151, 153)
        W_0_12 = np.random.rand(d_model_12)
        print(151, 154)
        emb_new_12 = np.zeros((tokens_12, d_model_12))
        for i_12 in range(tokens_12):
            print(153, 156)
            emb_new_12[i_12] = fc_layer(embeddings_12[i_12], W_12, W_0_12)
        print(154, 157)
        embeddings_12 = emb_new_12
        print(154, 158)
        embeddings_12 = arr_add(embeddings_12, emb_cpy_12)
        print(154, 159)
        embeddings_12 = BN_layer(embeddings_12, weights_12, biases_12)
    print('exit scope 12')


if __name__ == '__main__':
    main()
