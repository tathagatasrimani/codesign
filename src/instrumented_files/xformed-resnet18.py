import sys
from instrument_lib import *
import math
import numpy as np


def zero_pad_arr(img, zero_pad):
    print(1, 4)
    print(3, 5)
    len_new = len(img[0]) + 2 * zero_pad
    print(3, 6)
    wid_new = len(img[0][0]) + 2 * zero_pad
    print(3, 7)
    new_img = np.zeros((len(img), len_new, wid_new))
    print(4, 8)
    for i in range(len(img)):
        print(4, 8)
        print(5, 9)
        for j in range(len_new):
            print(5, 9)
            print(7, 10)
            make_zero = j < zero_pad or j >= len_new - zero_pad
            print(9, 11)
            for k in range(wid_new):
                print(9, 11)
                if k < zero_pad or k >= wid_new - zero_pad or make_zero:
                    print(10, 12)
                    print(12, 13)
                    new_img[i][j][k] = 0
                else:
                    print(10, 12)
                    print(14, 15)
                    new_img[i][j][k] = img[i][j - zero_pad][k - zero_pad]
    return new_img


def conv_layer(img, filt, numFilt, zero_pad, stride):
    print(1, 19)
    print(18, 20)
    f_len = int((len(img[0]) - len(filt[0]) + 2 * zero_pad) / stride + 1)
    print(18, 21)
    f_wid = int((len(img[0][0]) - len(filt[0][0]) + 2 * zero_pad) / stride + 1)
    print(18, 22)
    biases = np.random.rand(f_len)
    print(18, 23)
    img_new = zero_pad_arr(img, zero_pad)
    print(18, 24)
    f_new = np.zeros((numFilt, f_len, f_wid))
    print(19, 25)
    for i in range(numFilt):
        print(19, 25)
        print(20, 26)
        for j in range(f_len):
            print(20, 26)
            print(22, 27)
            for k in range(f_wid):
                print(22, 27)
                print(24, 28)
                sum_val = biases[j]
                print(26, 29)
                for l in range(len(filt)):
                    print(26, 29)
                    print(27, 30)
                    for c in range(len(filt[0])):
                        print(27, 30)
                        print(29, 31)
                        for d in range(len(filt[0][0])):
                            print(29, 31)
                            print(31, 32)
                            sum_val += img_new[l][j * stride + c][k *
                                stride + d] * filt[l][c][d]
                print(28, 33)
                f_new[i][j][k] = sum_val
    return f_new


def max_pool(input, l, w, zero_pad, stride):
    print(1, 36)
    if zero_pad > 0:
        print(36, 37)
        print(37, 38)
        input = zero_pad_arr(input, zero_pad)
    else:
        print(36, 37)
    print(38, 39)
    res_l = int((len(input[0]) - l) / stride + 1)
    print(38, 40)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    print(38, 41)
    result = np.zeros((len(input), res_l, res_w))
    print(39, 42)
    for i in range(len(input)):
        print(39, 42)
        print(40, 43)
        for j in range(res_l):
            print(40, 43)
            print(42, 44)
            for k in range(res_w):
                print(42, 44)
                print(44, 45)
                max_val = input[i][j][k]
                print(46, 46)
                for c in range(l):
                    print(46, 46)
                    print(47, 47)
                    for d in range(w):
                        print(47, 47)
                        print(49, 48)
                        max_val = max(max_val, input[i][j * stride + c][k *
                            stride + d])
                print(48, 49)
                result[i][j][k] = max_val / (l * w)
    return result


def avg_pool(input, l, w, zero_pad, stride):
    print(1, 52)
    if zero_pad > 0:
        print(54, 53)
        print(55, 54)
        input = zero_pad_arr(input, zero_pad)
    else:
        print(54, 53)
    print(56, 55)
    res_l = int((len(input[0]) - l) / stride + 1)
    print(56, 56)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    print(56, 57)
    result = np.zeros((len(input), res_l, res_w))
    print(57, 58)
    for i in range(len(input)):
        print(57, 58)
        print(58, 59)
        for j in range(res_l):
            print(58, 59)
            print(60, 60)
            for k in range(res_w):
                print(60, 60)
                print(62, 61)
                sum_val = 0
                print(64, 62)
                for c in range(l):
                    print(64, 62)
                    print(65, 63)
                    for d in range(w):
                        print(65, 63)
                        print(67, 64)
                        sum_val += input[i][j * stride + c][k * stride + d]
                print(66, 65)
                result[i][j][k] = sum_val / (l * w)
    return result


def reLU(img):
    print(1, 68)
    print(72, 69)
    for i in range(len(img)):
        print(72, 69)
        print(73, 70)
        for j in range(len(img[0])):
            print(73, 70)
            print(75, 71)
            for k in range(len(img[0][0])):
                print(75, 71)
                print(77, 72)
                img[i][j][k] = max(img[i][j][k], 0)
    return img


def get_mean(row):
    print(1, 75)
    print(82, 76)
    sum_val = 0
    print(83, 77)
    for i in range(len(row)):
        print(83, 77)
        print(84, 78)
        sum_val += row[i]
    return sum_val / len(row)


def std_dev(row):
    print(1, 81)
    print(89, 82)
    result = 0
    print(90, 83)
    for i in range(len(row)):
        print(90, 83)
        print(91, 84)
        diff = row[i] - get_mean(row)
        print(91, 85)
        result += diff * diff
    return math.sqrt(result / len(row))


def BN_layer(img, channels, weights, biases):
    print(1, 88)
    print(96, 89)
    for i in range(channels):
        print(96, 89)
        print(97, 90)
        for j in range(len(img[0])):
            print(97, 90)
            print(99, 91)
            dev = std_dev(img[i][j][:])
            print(99, 92)
            mean = get_mean(img[i][j][:])
            if dev == 0.0:
                print(101, 93)
                print(101, 93)
                dev = 1.0
            else:
                print(101, 93)
            print(102, 94)
            for k in range(len(img[0][0])):
                print(102, 94)
                print(103, 95)
                img[i][j][k] = weights[j] * ((img[i][j][k] - mean) / dev
                    ) + biases[j]
    return img


def flatten_layer(img):
    print(1, 98)
    print(108, 99)
    result = np.zeros(len(img) * len(img[0]) * len(img[0][0]))
    print(108, 100)
    index = 0
    print(109, 101)
    for i in range(len(img)):
        print(109, 101)
        print(110, 102)
        for j in range(len(img[0])):
            print(110, 102)
            print(112, 103)
            for k in range(len(img[0][0])):
                print(112, 103)
                print(114, 104)
                result[index] = img[i][j][k]
                print(114, 105)
                index += 1
    return result


def fc_layer(arr, W, W_0):
    print(1, 108)
    print(119, 109)
    result = np.zeros(len(W[0]))
    print(120, 110)
    for i in range(len(W[0])):
        print(120, 110)
        print(121, 111)
        sum_val = W_0[i]
        print(123, 112)
        for j in range(len(arr)):
            print(123, 112)
            print(124, 113)
            sum_val += arr[j] * W[j][i]
        print(125, 114)
        result[i] = sum_val
    return result


def softmax(img):
    print(1, 117)
    print(129, 118)
    sum_val = 0
    print(130, 119)
    for i in range(len(img)):
        print(130, 119)
        print(131, 120)
        sum_val += math.exp(img[i])
    print(132, 121)
    result = np.zeros(len(img))
    print(133, 122)
    for i in range(len(img)):
        print(133, 122)
        print(134, 123)
        result[i] = math.exp(img[i]) / sum_val
    return result


def main():
    print(1, 126)
    print(139, 127)
    zero_pad = 3
    print(139, 128)
    stride = 2
    print(139, 129)
    filt = np.random.rand(3, 7, 7)
    print(139, 130)
    img = np.random.rand(3, 64, 64)
    print(139, 133)
    img = conv_layer(img, filt, 1, zero_pad, stride)
    print(139, 134)
    weights = np.random.rand(len(img[0]))
    print(139, 135)
    biases = np.random.rand(len(img[0]))
    print(139, 136)
    img = BN_layer(img, 1, weights, biases)
    print(139, 137)
    img = reLU(img)
    print(139, 138)
    img = max_pool(img, 3, 3, 1, 2)
    print(139, 141)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(140, 142)
    for i in range(2):
        print(140, 142)
        print(141, 143)
        byPass = np.copy(img)
        print(141, 144)
        img = conv_layer(img, filt, 1, 3, 1)
        print(141, 145)
        img = BN_layer(img, 1, weights, biases)
        print(141, 146)
        img = reLU(img)
        print(141, 147)
        img = conv_layer(img, filt, 1, 3, 1)
        print(141, 148)
        img += byPass
        print(141, 149)
        img = reLU(img)
    print(142, 152)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(142, 153)
    byPass = np.copy(img)
    print(142, 154)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(142, 155)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(142, 156)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 157)
    img = BN_layer(img, 1, weights, biases)
    print(142, 158)
    img = reLU(img)
    print(142, 159)
    img = conv_layer(img, filt, 1, 3, 2)
    print(142, 160)
    img += byPass
    print(142, 161)
    img = reLU(img)
    print(142, 163)
    byPass = np.copy(img)
    print(142, 164)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 165)
    img = BN_layer(img, 1, weights, biases)
    print(142, 166)
    img = reLU(img)
    print(142, 167)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 168)
    img = BN_layer(img, 1, weights, biases)
    print(142, 169)
    img += byPass
    print(142, 170)
    img = reLU(img)
    print(142, 173)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(142, 174)
    byPass = np.copy(img)
    print(142, 175)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(142, 176)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(142, 177)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 178)
    img = BN_layer(img, 1, weights, biases)
    print(142, 179)
    img = reLU(img)
    print(142, 180)
    img = conv_layer(img, filt, 1, 3, 2)
    print(142, 181)
    img += byPass
    print(142, 182)
    img = reLU(img)
    print(142, 184)
    byPass = np.copy(img)
    print(142, 185)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 186)
    img = BN_layer(img, 1, weights, biases)
    print(142, 187)
    img = reLU(img)
    print(142, 188)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 189)
    img = BN_layer(img, 1, weights, biases)
    print(142, 190)
    img += byPass
    print(142, 191)
    img = reLU(img)
    print(142, 194)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(142, 195)
    byPass = np.copy(img)
    print(142, 196)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(142, 197)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(142, 198)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 199)
    img = BN_layer(img, 1, weights, biases)
    print(142, 200)
    img = reLU(img)
    print(142, 201)
    img = conv_layer(img, filt, 1, 3, 2)
    print(142, 202)
    img += byPass
    print(142, 203)
    img = reLU(img)
    print(142, 205)
    byPass = np.copy(img)
    print(142, 206)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 207)
    img = BN_layer(img, 1, weights, biases)
    print(142, 208)
    img = reLU(img)
    print(142, 209)
    img = conv_layer(img, filt, 1, 3, 1)
    print(142, 210)
    img = BN_layer(img, 1, weights, biases)
    print(142, 211)
    img += byPass
    print(142, 212)
    img = reLU(img)
    print(142, 213)
    img = avg_pool(img, len(img[0]), len(img[0][0]), 3, 1)
    print(142, 214)
    flat = flatten_layer(img)
    print(142, 216)
    weights = np.random.rand(len(img) * len(img[0]) * len(img[0][0]), 7)
    print(142, 217)
    w_0 = np.random.rand(7)
    print(142, 218)
    flat = fc_layer(flat, weights, w_0)
    print(142, 219)
    final = softmax(flat)
    print(final)
    return 0


if __name__ == '__main__':
    print(1, 224)
    main()
else:
    print(1, 224)
