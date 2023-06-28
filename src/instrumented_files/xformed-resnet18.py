import sys
import math
import numpy as np


class loop:
    print(1, 5)
    start_unroll = None
    print(1, 6)
    stop_unroll = None


def zero_pad_arr(img, zero_pad):
    print(1, 8)
    print(3, 9)
    len_new = len(img[0]) + 2 * zero_pad
    print(3, 10)
    wid_new = len(img[0][0]) + 2 * zero_pad
    print(3, 11)
    new_img = np.zeros((len(img), len_new, wid_new))
    for i in range(len(img)):
        for j in range(len_new):
            print(7, 14)
            make_zero = j < zero_pad or j >= len_new - zero_pad
            for k in range(wid_new):
                if k < zero_pad or k >= wid_new - zero_pad or make_zero:
                    print(10, 16)
                    print(12, 17)
                    new_img[i][j][k] = 0
                else:
                    print(10, 16)
                    print(14, 19)
                    new_img[i][j][k] = img[i][j - zero_pad][k - zero_pad]
    return new_img


def conv_layer(img, filt, numFilt, zero_pad, stride):
    print(1, 23)
    print(18, 24)
    f_len = int((len(img[0]) - len(filt[0]) + 2 * zero_pad) / stride + 1)
    print(18, 25)
    f_wid = int((len(img[0][0]) - len(filt[0][0]) + 2 * zero_pad) / stride + 1)
    print(18, 26)
    biases = np.random.rand(f_len)
    print(18, 27)
    img_new = zero_pad_arr(img, zero_pad)
    print(18, 28)
    f_new = np.zeros((numFilt, f_len, f_wid))
    for i in range(numFilt):
        for j in range(f_len):
            for k in range(f_wid):
                for l in range(len(filt)):
                    for c in range(len(filt[0])):
                        for d in range(len(filt[0][0])):
                            print(30, 35)
                            f_new[i][j][k] += img_new[l][j * stride + c][k *
                                stride + d] * filt[l][c][d]
    for i in range(numFilt):
        for j in range(f_len):
            for k in range(f_wid):
                print(36, 39)
                f_new[i][j][k] += biases[j]
    return f_new


def max_pool(input, l, w, zero_pad, stride):
    print(1, 42)
    if zero_pad > 0:
        print(41, 43)
        print(42, 44)
        input = zero_pad_arr(input, zero_pad)
    else:
        print(41, 43)
    print(43, 45)
    res_l = int((len(input[0]) - l) / stride + 1)
    print(43, 46)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    print(43, 47)
    result = np.zeros((len(input), res_l, res_w))
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                for c in range(l):
                    for d in range(w):
                        print(53, 53)
                        result[i][j][k] = max(input[i][j][k], result[i][j][
                            k], input[i][j * stride + c][k * stride + d])
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                print(59, 57)
                result[i][j][k] /= l * w
    return result


def avg_pool(input, l, w, zero_pad, stride):
    print(1, 60)
    if zero_pad > 0:
        print(64, 61)
        print(65, 62)
        input = zero_pad_arr(input, zero_pad)
    else:
        print(64, 61)
    print(66, 63)
    res_l = int((len(input[0]) - l) / stride + 1)
    print(66, 64)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    print(66, 65)
    result = np.zeros((len(input), res_l, res_w))
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                for c in range(l):
                    for d in range(w):
                        print(76, 71)
                        result[i][j][k] += input[i][j * stride + c][k *
                            stride + d]
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                print(82, 75)
                result[i][j][k] /= l * w
    return result


def reLU(img):
    print(1, 78)
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                print(92, 82)
                img[i][j][k] = max(img[i][j][k], 0)
    return img


def get_mean(row):
    print(1, 85)
    print(97, 86)
    sum_val = 0
    for i in range(len(row)):
        print(99, 88)
        sum_val += row[i]
    return sum_val / len(row)


def std_dev(row):
    print(1, 91)
    print(104, 92)
    result = 0
    for i in range(len(row)):
        print(106, 94)
        diff = row[i] - get_mean(row)
        print(106, 95)
        result += diff * diff
    return math.sqrt(result / len(row))


def BN_layer(img, channels, weights, biases):
    print(1, 98)
    for i in range(channels):
        for j in range(len(img[0])):
            print(114, 101)
            dev = std_dev(img[i][j][:])
            print(114, 102)
            mean = get_mean(img[i][j][:])
            if dev == 0.0:
                print(116, 103)
                print(116, 103)
                dev = 1.0
            else:
                print(116, 103)
            for k in range(len(img[0][0])):
                print(118, 105)
                img[i][j][k] = weights[j] * ((img[i][j][k] - mean) / dev
                    ) + biases[j]
    return img


def flatten_layer(img):
    print(1, 108)
    print(123, 109)
    result = np.zeros(len(img) * len(img[0]) * len(img[0][0]))
    print(123, 110)
    index = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                print(129, 114)
                result[index] = img[i][j][k]
                print(129, 115)
                index += 1
    return result


def fc_layer(arr, W, W_0):
    print(1, 118)
    print(134, 119)
    result = np.zeros(len(W[0]))
    for i in range(len(W[0])):
        print(136, 121)
        sum_val = W_0[i]
        for j in range(len(arr)):
            print(139, 123)
            sum_val += arr[j] * W[j][i]
        print(140, 124)
        result[i] = sum_val
    return result


def softmax(img):
    print(1, 127)
    print(144, 128)
    sum_val = 0
    for i in range(len(img)):
        print(146, 130)
        sum_val += math.exp(img[i])
    print(147, 131)
    result = np.zeros(len(img))
    for i in range(len(img)):
        print(149, 133)
        result[i] = math.exp(img[i]) / sum_val
    return result


def main():
    print(1, 136)
    print(154, 137)
    zero_pad = 3
    print(154, 138)
    stride = 2
    print(154, 139)
    filt = np.random.rand(3, 7, 7)
    print(154, 140)
    img = np.random.rand(3, 64, 64)
    print(154, 143)
    img = conv_layer(img, filt, 1, zero_pad, stride)
    print(154, 144)
    weights = np.random.rand(len(img[0]))
    print(154, 145)
    biases = np.random.rand(len(img[0]))
    print(154, 146)
    img = BN_layer(img, 1, weights, biases)
    print(154, 147)
    img = reLU(img)
    print(154, 148)
    img = max_pool(img, 3, 3, 1, 2)
    print(154, 151)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    for i in range(2):
        print(156, 153)
        byPass = np.copy(img)
        print(156, 154)
        img = conv_layer(img, filt, 1, 3, 1)
        print(156, 155)
        img = BN_layer(img, 1, weights, biases)
        print(156, 156)
        img = reLU(img)
        print(156, 157)
        img = conv_layer(img, filt, 1, 3, 1)
        print(156, 158)
        img += byPass
        print(156, 159)
        img = reLU(img)
    print(157, 162)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(157, 163)
    byPass = np.copy(img)
    print(157, 164)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(157, 165)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(157, 166)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 167)
    img = BN_layer(img, 1, weights, biases)
    print(157, 168)
    img = reLU(img)
    print(157, 169)
    img = conv_layer(img, filt, 1, 3, 2)
    print(157, 170)
    img += byPass
    print(157, 171)
    img = reLU(img)
    print(157, 173)
    byPass = np.copy(img)
    print(157, 174)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 175)
    img = BN_layer(img, 1, weights, biases)
    print(157, 176)
    img = reLU(img)
    print(157, 177)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 178)
    img = BN_layer(img, 1, weights, biases)
    print(157, 179)
    img += byPass
    print(157, 180)
    img = reLU(img)
    print(157, 183)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(157, 184)
    byPass = np.copy(img)
    print(157, 185)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(157, 186)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(157, 187)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 188)
    img = BN_layer(img, 1, weights, biases)
    print(157, 189)
    img = reLU(img)
    print(157, 190)
    img = conv_layer(img, filt, 1, 3, 2)
    print(157, 191)
    img += byPass
    print(157, 192)
    img = reLU(img)
    print(157, 194)
    byPass = np.copy(img)
    print(157, 195)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 196)
    img = BN_layer(img, 1, weights, biases)
    print(157, 197)
    img = reLU(img)
    print(157, 198)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 199)
    img = BN_layer(img, 1, weights, biases)
    print(157, 200)
    img += byPass
    print(157, 201)
    img = reLU(img)
    print(157, 204)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(157, 205)
    byPass = np.copy(img)
    print(157, 206)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(157, 207)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(157, 208)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 209)
    img = BN_layer(img, 1, weights, biases)
    print(157, 210)
    img = reLU(img)
    print(157, 211)
    img = conv_layer(img, filt, 1, 3, 2)
    print(157, 212)
    img += byPass
    print(157, 213)
    img = reLU(img)
    print(157, 215)
    byPass = np.copy(img)
    print(157, 216)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 217)
    img = BN_layer(img, 1, weights, biases)
    print(157, 218)
    img = reLU(img)
    print(157, 219)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 220)
    img = BN_layer(img, 1, weights, biases)
    print(157, 221)
    img += byPass
    print(157, 222)
    img = reLU(img)
    print(157, 223)
    img = avg_pool(img, len(img[0]), len(img[0][0]), 3, 1)
    print(157, 224)
    flat = flatten_layer(img)
    print(157, 226)
    weights = np.random.rand(len(img) * len(img[0]) * len(img[0][0]), 7)
    print(157, 227)
    w_0 = np.random.rand(7)
    print(157, 228)
    flat = fc_layer(flat, weights, w_0)
    print(157, 229)
    final = softmax(flat)
    return 0


if __name__ == '__main__':
    print(1, 233)
    loop.start_unroll
    main()
else:
    print(1, 233)
