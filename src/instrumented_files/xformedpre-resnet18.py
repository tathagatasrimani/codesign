import sys
from instrument_lib import *
import math
import numpy as np
from loop import loop


def zero_pad_arr(img, zero_pad):
    print('enter scope 1')
    print(1, 5)
    img__1 = img
    zero_pad__1 = zero_pad
    print(3, 6)
    len_new__1 = len(img__1[0]) + 2 * zero_pad__1
    print(3, 7)
    wid_new__1 = len(img__1[0][0]) + 2 * zero_pad__1
    print(3, 8)
    new_img__1 = np.zeros((len(img__1), len_new__1, wid_new__1))
    for i__1 in range(len(img__1)):
        print('enter scope 2')
        for j__2 in range(len_new__1):
            print('enter scope 3')
            print(7, 11)
            make_zero__3 = (j__2 < zero_pad__1 or j__2 >= len_new__1 -
                zero_pad__1)
            for k__3 in range(wid_new__1):
                print('enter scope 4')
                if (k__3 < zero_pad__1 or k__3 >= wid_new__1 - zero_pad__1 or
                    make_zero__3):
                    print(12, 14)
                    new_img__1[i__1][j__2][k__3] = 0
                else:
                    print(14, 16)
                    new_img__1[i__1][j__2][k__3] = img__1[i__1][j__2 -
                        zero_pad__1][k__3 - zero_pad__1]
                print('exit scope 4')
            print('exit scope 3')
        print('exit scope 2')
    print('exit scope 1')
    return new_img__1
    print('exit scope 1')


def conv_layer(img, filt, numFilt, zero_pad, stride):
    print('enter scope 5')
    print(1, 20)
    img__5 = img
    filt__5 = filt
    numFilt__5 = numFilt
    zero_pad__5 = zero_pad
    stride__5 = stride
    print(18, 21)
    f_len__5 = int((len(img__5[0]) - len(filt__5[0]) + 2 * zero_pad__5) /
        stride__5 + 1)
    print(18, 22)
    f_wid__5 = int((len(img__5[0][0]) - len(filt__5[0][0]) + 2 *
        zero_pad__5) / stride__5 + 1)
    print(18, 23)
    biases__5 = np.random.rand(f_len__5)
    print(18, 24)
    img_new__5 = zero_pad_arr(img__5, zero_pad__5)
    print(18, 25)
    f_new__5 = np.zeros((numFilt__5, f_len__5, f_wid__5))
    for i__5 in range(numFilt__5):
        print('enter scope 6')
        for j__6 in range(f_len__5):
            print('enter scope 7')
            for k__7 in range(f_wid__5):
                print('enter scope 8')
                for l__8 in range(len(filt__5)):
                    print('enter scope 9')
                    for c__9 in range(len(filt__5[0])):
                        print('enter scope 10')
                        for d__10 in range(len(filt__5[0][0])):
                            print('enter scope 11')
                            print(30, 32)
                            f_new__5[i__5][j__6][k__7] += img_new__5[l__8][
                                j__6 * stride__5 + c__9][k__7 * stride__5 +
                                d__10] * filt__5[l__8][c__9][d__10]
                            print('exit scope 11')
                        print('exit scope 10')
                    print('exit scope 9')
                print('exit scope 8')
            print('exit scope 7')
        print('exit scope 6')
    for i__5 in range(numFilt__5):
        print('enter scope 12')
        for j__12 in range(f_len__5):
            print('enter scope 13')
            for k__13 in range(f_wid__5):
                print('enter scope 14')
                print(36, 36)
                f_new__5[i__5][j__12][k__13] += biases__5[j__12]
                print('exit scope 14')
            print('exit scope 13')
        print('exit scope 12')
    print('exit scope 5')
    return f_new__5
    print('exit scope 5')


def max_pool(input, l, w, zero_pad, stride):
    print('enter scope 15')
    print(1, 39)
    input__15 = input
    l__15 = l
    w__15 = w
    zero_pad__15 = zero_pad
    stride__15 = stride
    if zero_pad__15 > 0:
        print(42, 41)
        input__15 = zero_pad_arr(input__15, zero_pad__15)
    print(43, 42)
    res_l__15 = int((len(input__15[0]) - l__15) / stride__15 + 1)
    print(43, 43)
    res_w__15 = int((len(input__15[0][0]) - w__15) / stride__15 + 1)
    print(43, 44)
    result__15 = np.zeros((len(input__15), res_l__15, res_w__15))
    for i__15 in range(len(input__15)):
        print('enter scope 16')
        for j__16 in range(res_l__15):
            print('enter scope 17')
            for k__17 in range(res_w__15):
                print('enter scope 18')
                for c__18 in range(l__15):
                    print('enter scope 19')
                    for d__19 in range(w__15):
                        print('enter scope 20')
                        print(53, 50)
                        result__15[i__15][j__16][k__17] = max(input__15[
                            i__15][j__16][k__17], result__15[i__15][j__16][
                            k__17], input__15[i__15][j__16 * stride__15 +
                            c__18][k__17 * stride__15 + d__19])
                        print('exit scope 20')
                    print('exit scope 19')
                print('exit scope 18')
            print('exit scope 17')
        print('exit scope 16')
    for i__15 in range(len(input__15)):
        print('enter scope 21')
        for j__21 in range(res_l__15):
            print('enter scope 22')
            for k__22 in range(res_w__15):
                print('enter scope 23')
                print(59, 54)
                result__15[i__15][j__21][k__22] /= l__15 * w__15
                print('exit scope 23')
            print('exit scope 22')
        print('exit scope 21')
    print('exit scope 15')
    return result__15
    print('exit scope 15')


def avg_pool(input, l, w, zero_pad, stride):
    print('enter scope 24')
    print(1, 57)
    input__24 = input
    l__24 = l
    w__24 = w
    zero_pad__24 = zero_pad
    stride__24 = stride
    if zero_pad__24 > 0:
        print(65, 59)
        input__24 = zero_pad_arr(input__24, zero_pad__24)
    print(66, 60)
    res_l__24 = int((len(input__24[0]) - l__24) / stride__24 + 1)
    print(66, 61)
    res_w__24 = int((len(input__24[0][0]) - w__24) / stride__24 + 1)
    print(66, 62)
    result__24 = np.zeros((len(input__24), res_l__24, res_w__24))
    for i__24 in range(len(input__24)):
        print('enter scope 25')
        for j__25 in range(res_l__24):
            print('enter scope 26')
            for k__26 in range(res_w__24):
                print('enter scope 27')
                for c__27 in range(l__24):
                    print('enter scope 28')
                    for d__28 in range(w__24):
                        print('enter scope 29')
                        print(76, 68)
                        result__24[i__24][j__25][k__26] += input__24[i__24][
                            j__25 * stride__24 + c__27][k__26 * stride__24 +
                            d__28]
                        print('exit scope 29')
                    print('exit scope 28')
                print('exit scope 27')
            print('exit scope 26')
        print('exit scope 25')
    for i__24 in range(len(input__24)):
        print('enter scope 30')
        for j__30 in range(res_l__24):
            print('enter scope 31')
            for k__31 in range(res_w__24):
                print('enter scope 32')
                print(82, 72)
                result__24[i__24][j__30][k__31] /= l__24 * w__24
                print('exit scope 32')
            print('exit scope 31')
        print('exit scope 30')
    print('exit scope 24')
    return result__24
    print('exit scope 24')


def reLU(img):
    print('enter scope 33')
    print(1, 75)
    img__33 = img
    for i__33 in range(len(img__33)):
        print('enter scope 34')
        for j__34 in range(len(img__33[0])):
            print('enter scope 35')
            for k__35 in range(len(img__33[0][0])):
                print('enter scope 36')
                print(92, 79)
                img__33[i__33][j__34][k__35] = max(img__33[i__33][j__34][
                    k__35], 0)
                print('exit scope 36')
            print('exit scope 35')
        print('exit scope 34')
    print('exit scope 33')
    return img__33
    print('exit scope 33')


def get_mean(row):
    print('enter scope 37')
    print(1, 82)
    row__37 = row
    print(97, 83)
    sum_val__37 = 0
    for i__37 in range(len(row__37)):
        print('enter scope 38')
        print(99, 85)
        sum_val__37 += row__37[i__37]
        print('exit scope 38')
    print('exit scope 37')
    return sum_val__37 / len(row__37)
    print('exit scope 37')


def std_dev(row):
    print('enter scope 39')
    print(1, 88)
    row__39 = row
    print(104, 89)
    result__39 = 0
    for i__39 in range(len(row__39)):
        print('enter scope 40')
        print(106, 91)
        diff__40 = row__39[i__39] - get_mean(row__39)
        print(106, 92)
        result__39 += diff__40 * diff__40
        print('exit scope 40')
    print('exit scope 39')
    return math.sqrt(result__39 / len(row__39))
    print('exit scope 39')


def BN_layer(img, channels, weights, biases):
    print('enter scope 41')
    print(1, 95)
    img__41 = img
    channels__41 = channels
    weights__41 = weights
    biases__41 = biases
    for i__41 in range(channels__41):
        print('enter scope 42')
        for j__42 in range(len(img__41[0])):
            print('enter scope 43')
            print(114, 98)
            dev__43 = std_dev(img__41[i__41][j__42][:])
            print(114, 99)
            mean__43 = get_mean(img__41[i__41][j__42][:])
            if dev__43 == 0.0:
                print(116, 100)
                dev__43 = 1.0
            for k__43 in range(len(img__41[0][0])):
                print('enter scope 44')
                print(118, 102)
                img__41[i__41][j__42][k__43] = weights__41[j__42] * ((
                    img__41[i__41][j__42][k__43] - mean__43) / dev__43
                    ) + biases__41[j__42]
                print('exit scope 44')
            print('exit scope 43')
        print('exit scope 42')
    print('exit scope 41')
    return img__41
    print('exit scope 41')


def flatten_layer(img):
    print('enter scope 45')
    print(1, 105)
    img__45 = img
    print(123, 106)
    result__45 = np.zeros(len(img__45) * len(img__45[0]) * len(img__45[0][0]))
    print(123, 107)
    index__45 = 0
    for i__45 in range(len(img__45)):
        print('enter scope 46')
        for j__46 in range(len(img__45[0])):
            print('enter scope 47')
            for k__47 in range(len(img__45[0][0])):
                print('enter scope 48')
                print(129, 111)
                result__45[index__45] = img__45[i__45][j__46][k__47]
                print(129, 112)
                index__45 += 1
                print('exit scope 48')
            print('exit scope 47')
        print('exit scope 46')
    print('exit scope 45')
    return result__45
    print('exit scope 45')


def fc_layer(arr, W, W_0):
    print('enter scope 49')
    print(1, 115)
    arr__49 = arr
    W__49 = W
    W_0__49 = W_0
    print(134, 116)
    result__49 = np.zeros(len(W__49[0]))
    for i__49 in range(len(W__49[0])):
        print('enter scope 50')
        print(136, 118)
        sum_val__50 = W_0__49[i__49]
        for j__50 in range(len(arr__49)):
            print('enter scope 51')
            print(139, 120)
            sum_val__50 += arr__49[j__50] * W__49[j__50][i__49]
            print('exit scope 51')
        print(140, 121)
        result__49[i__49] = sum_val__50
        print('exit scope 50')
    print('exit scope 49')
    return result__49
    print('exit scope 49')


def softmax(img):
    print('enter scope 52')
    print(1, 124)
    img__52 = img
    print(144, 125)
    sum_val__52 = 0
    for i__52 in range(len(img__52)):
        print('enter scope 53')
        print(146, 127)
        sum_val__52 += math.exp(img__52[i__52])
        print('exit scope 53')
    print(147, 128)
    result__52 = np.zeros(len(img__52))
    for i__52 in range(len(img__52)):
        print('enter scope 54')
        print(149, 130)
        result__52[i__52] = math.exp(img__52[i__52]) / sum_val__52
        print('exit scope 54')
    print('exit scope 52')
    return result__52
    print('exit scope 52')


def main():
    print('enter scope 55')
    print(1, 133)
    print(154, 134)
    zero_pad__55 = 3
    print(154, 135)
    stride__55 = 2
    print(154, 136)
    filt__55 = np.random.rand(3, 7, 7)
    print(154, 137)
    img__55 = np.random.rand(3, 64, 64)
    print(154, 140)
    img__55 = conv_layer(img__55, filt__55, 1, zero_pad__55, stride__55)
    print(154, 141)
    weights__55 = np.random.rand(len(img__55[0]))
    print(154, 142)
    biases__55 = np.random.rand(len(img__55[0]))
    print(154, 143)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(154, 144)
    img__55 = reLU(img__55)
    print(154, 145)
    img__55 = max_pool(img__55, 3, 3, 1, 2)
    print(154, 148)
    filt__55 = np.random.rand(1, len(filt__55[0]), len(filt__55[0][0]))
    for i__55 in range(2):
        print('enter scope 56')
        print(156, 150)
        byPass__56 = np.copy(img__55)
        print(156, 151)
        img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
        print(156, 152)
        img__55 = BN_layer(img__55, 1, weights__55, biases__55)
        print(156, 153)
        img__55 = reLU(img__55)
        print(156, 154)
        img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
        print(156, 155)
        img__55 += byPass__56
        print(156, 156)
        img__55 = reLU(img__55)
        print('exit scope 56')
    print(157, 159)
    filt__55 = np.random.rand(1, len(filt__55[0]), len(filt__55[0][0]))
    print(157, 160)
    byPass__55 = np.copy(img__55)
    print(157, 161)
    byPass__55 = conv_layer(byPass__55, filt__55, 1, 3, 2)
    print(157, 162)
    byPass__55 = BN_layer(byPass__55, 1, weights__55, biases__55)
    print(157, 163)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 164)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 165)
    img__55 = reLU(img__55)
    print(157, 166)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 2)
    print(157, 167)
    img__55 += byPass__55
    print(157, 168)
    img__55 = reLU(img__55)
    print(157, 170)
    byPass__55 = np.copy(img__55)
    print(157, 171)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 172)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 173)
    img__55 = reLU(img__55)
    print(157, 174)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 175)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 176)
    img__55 += byPass__55
    print(157, 177)
    img__55 = reLU(img__55)
    print(157, 180)
    filt__55 = np.random.rand(1, len(filt__55[0]), len(filt__55[0][0]))
    print(157, 181)
    byPass__55 = np.copy(img__55)
    print(157, 182)
    byPass__55 = conv_layer(byPass__55, filt__55, 1, 3, 2)
    print(157, 183)
    byPass__55 = BN_layer(byPass__55, 1, weights__55, biases__55)
    print(157, 184)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 185)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 186)
    img__55 = reLU(img__55)
    print(157, 187)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 2)
    print(157, 188)
    img__55 += byPass__55
    print(157, 189)
    img__55 = reLU(img__55)
    print(157, 191)
    byPass__55 = np.copy(img__55)
    print(157, 192)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 193)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 194)
    img__55 = reLU(img__55)
    print(157, 195)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 196)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 197)
    img__55 += byPass__55
    print(157, 198)
    img__55 = reLU(img__55)
    print(157, 201)
    filt__55 = np.random.rand(1, len(filt__55[0]), len(filt__55[0][0]))
    print(157, 202)
    byPass__55 = np.copy(img__55)
    print(157, 203)
    byPass__55 = conv_layer(byPass__55, filt__55, 1, 3, 2)
    print(157, 204)
    byPass__55 = BN_layer(byPass__55, 1, weights__55, biases__55)
    print(157, 205)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 206)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 207)
    img__55 = reLU(img__55)
    print(157, 208)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 2)
    print(157, 209)
    img__55 += byPass__55
    print(157, 210)
    img__55 = reLU(img__55)
    print(157, 212)
    byPass__55 = np.copy(img__55)
    print(157, 213)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 214)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 215)
    img__55 = reLU(img__55)
    print(157, 216)
    img__55 = conv_layer(img__55, filt__55, 1, 3, 1)
    print(157, 217)
    img__55 = BN_layer(img__55, 1, weights__55, biases__55)
    print(157, 218)
    img__55 += byPass__55
    print(157, 219)
    img__55 = reLU(img__55)
    print(157, 220)
    img__55 = avg_pool(img__55, len(img__55[0]), len(img__55[0][0]), 3, 1)
    print(157, 221)
    flat__55 = flatten_layer(img__55)
    print(157, 223)
    weights__55 = np.random.rand(len(img__55) * len(img__55[0]) * len(
        img__55[0][0]), 7)
    print(157, 224)
    w_0__55 = np.random.rand(7)
    print(157, 225)
    flat__55 = fc_layer(flat__55, weights__55, w_0__55)
    print(157, 226)
    final__55 = softmax(flat__55)
    print('exit scope 55')
    return 0
    print('exit scope 55')


if __name__ == '__main__':
    loop.start_unroll
    main()
