import sys
from instrument_lib import *
import math
import numpy as np
from loop import loop


def zero_pad_arr(img, zero_pad):
    print('enter scope 1')
    print(1, 5)
    img_1 = img
    zero_pad_1 = zero_pad
    print(3, 6)
    len_new_1 = len(img_1[0]) + 2 * zero_pad_1
    print(3, 7)
    wid_new_1 = len(img_1[0][0]) + 2 * zero_pad_1
    print(3, 8)
    new_img_1 = np.zeros((len(img_1), len_new_1, wid_new_1))
    for i_1 in range(len(img_1)):
        for j_1 in range(len_new_1):
            print(7, 11)
            make_zero_1 = j_1 < zero_pad_1 or j_1 >= len_new_1 - zero_pad_1
            for k_1 in range(wid_new_1):
                if (k_1 < zero_pad_1 or k_1 >= wid_new_1 - zero_pad_1 or
                    make_zero_1):
                    print(12, 14)
                    new_img_1[i_1][j_1][k_1] = 0
                else:
                    print(14, 16)
                    new_img_1[i_1][j_1][k_1] = img_1[i_1][j_1 - zero_pad_1][
                        k_1 - zero_pad_1]
    print('exit scope 1')
    return new_img_1
    print('exit scope 1')


def conv_layer(img, filt, numFilt, zero_pad, stride):
    print('enter scope 2')
    print(1, 20)
    img_2 = img
    filt_2 = filt
    numFilt_2 = numFilt
    zero_pad_2 = zero_pad
    stride_2 = stride
    print(18, 21)
    f_len_2 = int((len(img_2[0]) - len(filt_2[0]) + 2 * zero_pad_2) /
        stride_2 + 1)
    print(18, 22)
    f_wid_2 = int((len(img_2[0][0]) - len(filt_2[0][0]) + 2 * zero_pad_2) /
        stride_2 + 1)
    print(18, 23)
    biases_2 = np.random.rand(f_len_2)
    print(18, 24)
    img_new_2 = zero_pad_arr(img_2, zero_pad_2)
    print(18, 25)
    f_new_2 = np.zeros((numFilt_2, f_len_2, f_wid_2))
    for i_2 in range(numFilt_2):
        for j_2 in range(f_len_2):
            for k_2 in range(f_wid_2):
                for l_2 in range(len(filt_2)):
                    for c_2 in range(len(filt_2[0])):
                        for d_2 in range(len(filt_2[0][0])):
                            print(30, 32)
                            f_new_2[i_2][j_2][k_2] += img_new_2[l_2][j_2 *
                                stride_2 + c_2][k_2 * stride_2 + d_2] * filt_2[
                                l_2][c_2][d_2]
    for i_2 in range(numFilt_2):
        for j_2 in range(f_len_2):
            for k_2 in range(f_wid_2):
                print(36, 36)
                f_new_2[i_2][j_2][k_2] += biases_2[j_2]
    print('exit scope 2')
    return f_new_2
    print('exit scope 2')


def max_pool(input, l, w, zero_pad, stride):
    print('enter scope 3')
    print(1, 39)
    input_3 = input
    l_3 = l
    w_3 = w
    zero_pad_3 = zero_pad
    stride_3 = stride
    if zero_pad_3 > 0:
        print(42, 41)
        input_3 = zero_pad_arr(input_3, zero_pad_3)
    print(43, 42)
    res_l_3 = int((len(input_3[0]) - l_3) / stride_3 + 1)
    print(43, 43)
    res_w_3 = int((len(input_3[0][0]) - w_3) / stride_3 + 1)
    print(43, 44)
    result_3 = np.zeros((len(input_3), res_l_3, res_w_3))
    for i_3 in range(len(input_3)):
        for j_3 in range(res_l_3):
            for k_3 in range(res_w_3):
                for c_3 in range(l_3):
                    for d_3 in range(w_3):
                        print(53, 50)
                        result_3[i_3][j_3][k_3] = max(input_3[i_3][j_3][k_3
                            ], result_3[i_3][j_3][k_3], input_3[i_3][j_3 *
                            stride_3 + c_3][k_3 * stride_3 + d_3])
    for i_3 in range(len(input_3)):
        for j_3 in range(res_l_3):
            for k_3 in range(res_w_3):
                print(59, 54)
                result_3[i_3][j_3][k_3] /= l_3 * w_3
    print('exit scope 3')
    return result_3
    print('exit scope 3')


def avg_pool(input, l, w, zero_pad, stride):
    print('enter scope 4')
    print(1, 57)
    input_4 = input
    l_4 = l
    w_4 = w
    zero_pad_4 = zero_pad
    stride_4 = stride
    if zero_pad_4 > 0:
        print(65, 59)
        input_4 = zero_pad_arr(input_4, zero_pad_4)
    print(66, 60)
    res_l_4 = int((len(input_4[0]) - l_4) / stride_4 + 1)
    print(66, 61)
    res_w_4 = int((len(input_4[0][0]) - w_4) / stride_4 + 1)
    print(66, 62)
    result_4 = np.zeros((len(input_4), res_l_4, res_w_4))
    for i_4 in range(len(input_4)):
        for j_4 in range(res_l_4):
            for k_4 in range(res_w_4):
                for c_4 in range(l_4):
                    for d_4 in range(w_4):
                        print(76, 68)
                        result_4[i_4][j_4][k_4] += input_4[i_4][j_4 *
                            stride_4 + c_4][k_4 * stride_4 + d_4]
    for i_4 in range(len(input_4)):
        for j_4 in range(res_l_4):
            for k_4 in range(res_w_4):
                print(82, 72)
                result_4[i_4][j_4][k_4] /= l_4 * w_4
    print('exit scope 4')
    return result_4
    print('exit scope 4')


def reLU(img):
    print('enter scope 5')
    print(1, 75)
    img_5 = img
    for i_5 in range(len(img_5)):
        for j_5 in range(len(img_5[0])):
            for k_5 in range(len(img_5[0][0])):
                print(92, 79)
                img_5[i_5][j_5][k_5] = max(img_5[i_5][j_5][k_5], 0)
    print('exit scope 5')
    return img_5
    print('exit scope 5')


def get_mean(row):
    print('enter scope 6')
    print(1, 82)
    row_6 = row
    print(97, 83)
    sum_val_6 = 0
    for i_6 in range(len(row_6)):
        print(99, 85)
        sum_val_6 += row_6[i_6]
    print('exit scope 6')
    return sum_val_6 / len(row_6)
    print('exit scope 6')


def std_dev(row):
    print('enter scope 7')
    print(1, 88)
    row_7 = row
    print(104, 89)
    result_7 = 0
    for i_7 in range(len(row_7)):
        print(106, 91)
        diff_7 = row_7[i_7] - get_mean(row_7)
        print(106, 92)
        result_7 += diff_7 * diff_7
    print('exit scope 7')
    return math.sqrt(result_7 / len(row_7))
    print('exit scope 7')


def BN_layer(img, channels, weights, biases):
    print('enter scope 8')
    print(1, 95)
    img_8 = img
    channels_8 = channels
    weights_8 = weights
    biases_8 = biases
    for i_8 in range(channels_8):
        for j_8 in range(len(img_8[0])):
            print(114, 98)
            dev_8 = std_dev(img_8[i_8][j_8][:])
            print(114, 99)
            mean_8 = get_mean(img_8[i_8][j_8][:])
            if dev_8 == 0.0:
                print(116, 100)
                dev_8 = 1.0
            for k_8 in range(len(img_8[0][0])):
                print(118, 102)
                img_8[i_8][j_8][k_8] = weights_8[j_8] * ((img_8[i_8][j_8][
                    k_8] - mean_8) / dev_8) + biases_8[j_8]
    print('exit scope 8')
    return img_8
    print('exit scope 8')


def flatten_layer(img):
    print('enter scope 9')
    print(1, 105)
    img_9 = img
    print(123, 106)
    result_9 = np.zeros(len(img_9) * len(img_9[0]) * len(img_9[0][0]))
    print(123, 107)
    index_9 = 0
    for i_9 in range(len(img_9)):
        for j_9 in range(len(img_9[0])):
            for k_9 in range(len(img_9[0][0])):
                print(129, 111)
                result_9[index_9] = img_9[i_9][j_9][k_9]
                print(129, 112)
                index_9 += 1
    print('exit scope 9')
    return result_9
    print('exit scope 9')


def fc_layer(arr, W, W_0):
    print('enter scope 10')
    print(1, 115)
    arr_10 = arr
    W_10 = W
    W_0_10 = W_0
    print(134, 116)
    result_10 = np.zeros(len(W_10[0]))
    for i_10 in range(len(W_10[0])):
        print(136, 118)
        sum_val_10 = W_0_10[i_10]
        for j_10 in range(len(arr_10)):
            print(139, 120)
            sum_val_10 += arr_10[j_10] * W_10[j_10][i_10]
        print(140, 121)
        result_10[i_10] = sum_val_10
    print('exit scope 10')
    return result_10
    print('exit scope 10')


def softmax(img):
    print('enter scope 11')
    print(1, 124)
    img_11 = img
    print(144, 125)
    sum_val_11 = 0
    for i_11 in range(len(img_11)):
        print(146, 127)
        sum_val_11 += math.exp(img_11[i_11])
    print(147, 128)
    result_11 = np.zeros(len(img_11))
    for i_11 in range(len(img_11)):
        print(149, 130)
        result_11[i_11] = math.exp(img_11[i_11]) / sum_val_11
    print('exit scope 11')
    return result_11
    print('exit scope 11')


def main():
    print('enter scope 12')
    print(1, 133)
    print(154, 134)
    zero_pad_12 = 3
    print(154, 135)
    stride_12 = 2
    print(154, 136)
    filt_12 = np.random.rand(3, 7, 7)
    print(154, 137)
    img_12 = np.random.rand(3, 224, 224)
    print(154, 140)
    img_12 = conv_layer(img_12, filt_12, 1, zero_pad_12, stride_12)
    print(154, 141)
    weights_12 = np.random.rand(len(img_12[0]))
    print(154, 142)
    biases_12 = np.random.rand(len(img_12[0]))
    print(154, 143)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(154, 144)
    img_12 = reLU(img_12)
    print(154, 145)
    img_12 = max_pool(img_12, 3, 3, 1, 2)
    print(154, 148)
    filt_12 = np.random.rand(1, len(filt_12[0]), len(filt_12[0][0]))
    for i_12 in range(2):
        print(156, 150)
        byPass_12 = np.copy(img_12)
        print(156, 151)
        img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
        print(156, 152)
        img_12 = BN_layer(img_12, 1, weights_12, biases_12)
        print(156, 153)
        img_12 = reLU(img_12)
        print(156, 154)
        img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
        print(156, 155)
        img_12 += byPass_12
        print(156, 156)
        img_12 = reLU(img_12)
    print(157, 159)
    filt_12 = np.random.rand(1, len(filt_12[0]), len(filt_12[0][0]))
    print(157, 160)
    byPass_12 = np.copy(img_12)
    print(157, 161)
    byPass_12 = conv_layer(byPass_12, filt_12, 1, 3, 2)
    print(157, 162)
    byPass_12 = BN_layer(byPass_12, 1, weights_12, biases_12)
    print(157, 163)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 164)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 165)
    img_12 = reLU(img_12)
    print(157, 166)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 2)
    print(157, 167)
    img_12 += byPass_12
    print(157, 168)
    img_12 = reLU(img_12)
    print(157, 170)
    byPass_12 = np.copy(img_12)
    print(157, 171)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 172)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 173)
    img_12 = reLU(img_12)
    print(157, 174)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 175)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 176)
    img_12 += byPass_12
    print(157, 177)
    img_12 = reLU(img_12)
    print(157, 180)
    filt_12 = np.random.rand(1, len(filt_12[0]), len(filt_12[0][0]))
    print(157, 181)
    byPass_12 = np.copy(img_12)
    print(157, 182)
    byPass_12 = conv_layer(byPass_12, filt_12, 1, 3, 2)
    print(157, 183)
    byPass_12 = BN_layer(byPass_12, 1, weights_12, biases_12)
    print(157, 184)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 185)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 186)
    img_12 = reLU(img_12)
    print(157, 187)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 2)
    print(157, 188)
    img_12 += byPass_12
    print(157, 189)
    img_12 = reLU(img_12)
    print(157, 191)
    byPass_12 = np.copy(img_12)
    print(157, 192)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 193)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 194)
    img_12 = reLU(img_12)
    print(157, 195)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 196)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 197)
    img_12 += byPass_12
    print(157, 198)
    img_12 = reLU(img_12)
    print(157, 201)
    filt_12 = np.random.rand(1, len(filt_12[0]), len(filt_12[0][0]))
    print(157, 202)
    byPass_12 = np.copy(img_12)
    print(157, 203)
    byPass_12 = conv_layer(byPass_12, filt_12, 1, 3, 2)
    print(157, 204)
    byPass_12 = BN_layer(byPass_12, 1, weights_12, biases_12)
    print(157, 205)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 206)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 207)
    img_12 = reLU(img_12)
    print(157, 208)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 2)
    print(157, 209)
    img_12 += byPass_12
    print(157, 210)
    img_12 = reLU(img_12)
    print(157, 212)
    byPass_12 = np.copy(img_12)
    print(157, 213)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 214)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 215)
    img_12 = reLU(img_12)
    print(157, 216)
    img_12 = conv_layer(img_12, filt_12, 1, 3, 1)
    print(157, 217)
    img_12 = BN_layer(img_12, 1, weights_12, biases_12)
    print(157, 218)
    img_12 += byPass_12
    print(157, 219)
    img_12 = reLU(img_12)
    print(157, 220)
    img_12 = avg_pool(img_12, len(img_12[0]), len(img_12[0][0]), 3, 1)
    print(157, 221)
    flat_12 = flatten_layer(img_12)
    print(157, 223)
    weights_12 = np.random.rand(len(img_12) * len(img_12[0]) * len(img_12[0
        ][0]), 7)
    print(157, 224)
    w_0_12 = np.random.rand(7)
    print(157, 225)
    flat_12 = fc_layer(flat_12, weights_12, w_0_12)
    print(157, 226)
    final_12 = softmax(flat_12)
    print('exit scope 12')
    return 0
    print('exit scope 12')


if __name__ == '__main__':
    loop.start_unroll
    main()
