import sys
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
    for i in range(len(img)):
        for j in range(len_new):
            print(7, 10)
            make_zero = j < zero_pad or j >= len_new - zero_pad
            for k in range(wid_new):
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
    for i in range(numFilt):
        for j in range(f_len):
            for k in range(f_wid):
                for l in range(len(filt)):
                    for c in range(len(filt[0])):
                        for d in range(len(filt[0][0])):
                            print(30, 31)
                            f_new[i][j][k] += img_new[l][j * stride + c][k *
                                stride + d] * filt[l][c][d]
    for i in range(numFilt):
        for j in range(f_len):
            for k in range(f_wid):
                print(36, 35)
                f_new[i][j][k] += biases[j]
    return f_new


def max_pool(input, l, w, zero_pad, stride):
    print(1, 38)
    if zero_pad > 0:
        print(41, 39)
        print(42, 40)
        input = zero_pad_arr(input, zero_pad)
    else:
        print(41, 39)
    print(43, 41)
    res_l = int((len(input[0]) - l) / stride + 1)
    print(43, 42)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    print(43, 43)
    result = np.zeros((len(input), res_l, res_w))
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                for c in range(l):
                    for d in range(w):
                        print(53, 49)
                        result[i][j][k] = max(input[i][j][k], result[i][j][
                            k], input[i][j * stride + c][k * stride + d])
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                print(59, 53)
                result[i][j][k] /= l * w
    return result


def avg_pool(input, l, w, zero_pad, stride):
    print(1, 56)
    if zero_pad > 0:
        print(64, 57)
        print(65, 58)
        input = zero_pad_arr(input, zero_pad)
    else:
        print(64, 57)
    print(66, 59)
    res_l = int((len(input[0]) - l) / stride + 1)
    print(66, 60)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    print(66, 61)
    result = np.zeros((len(input), res_l, res_w))
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                for c in range(l):
                    for d in range(w):
                        print(76, 67)
                        result[i][j][k] += input[i][j * stride + c][k *
                            stride + d]
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                print(82, 71)
                result[i][j][k] /= l * w
    return result


def reLU(img):
    print(1, 74)
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                print(92, 78)
                img[i][j][k] = max(img[i][j][k], 0)
    return img


def get_mean(row):
    print(1, 81)
    print(97, 82)
    sum_val = 0
    for i in range(len(row)):
        print(99, 84)
        sum_val += row[i]
    return sum_val / len(row)


def std_dev(row):
    print(1, 87)
    print(104, 88)
    result = 0
    for i in range(len(row)):
        print(106, 90)
        diff = row[i] - get_mean(row)
        print(106, 91)
        result += diff * diff
    return math.sqrt(result / len(row))


def BN_layer(img, channels, weights, biases):
    print(1, 94)
    for i in range(channels):
        for j in range(len(img[0])):
            print(114, 97)
            dev = std_dev(img[i][j][:])
            print(114, 98)
            mean = get_mean(img[i][j][:])
            if dev == 0.0:
                print(116, 99)
                print(116, 99)
                dev = 1.0
            else:
                print(116, 99)
            for k in range(len(img[0][0])):
                print(118, 101)
                img[i][j][k] = weights[j] * ((img[i][j][k] - mean) / dev
                    ) + biases[j]
    return img


def flatten_layer(img):
    print(1, 104)
    print(123, 105)
    result = np.zeros(len(img) * len(img[0]) * len(img[0][0]))
    print(123, 106)
    index = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                print(129, 110)
                result[index] = img[i][j][k]
                print(129, 111)
                index += 1
    return result


def fc_layer(arr, W, W_0):
    print(1, 114)
    print(134, 115)
    result = np.zeros(len(W[0]))
    for i in range(len(W[0])):
        print(136, 117)
        sum_val = W_0[i]
        for j in range(len(arr)):
            print(139, 119)
            sum_val += arr[j] * W[j][i]
        print(140, 120)
        result[i] = sum_val
    return result


def softmax(img):
    print(1, 123)
    print(144, 124)
    sum_val = 0
    for i in range(len(img)):
        print(146, 126)
        sum_val += math.exp(img[i])
    print(147, 127)
    result = np.zeros(len(img))
    for i in range(len(img)):
        print(149, 129)
        result[i] = math.exp(img[i]) / sum_val
    return result


def main():
    print(1, 132)
    print(154, 133)
    zero_pad = 3
    print(154, 134)
    stride = 2
    print(154, 135)
    filt = np.random.rand(3, 7, 7)
    print(154, 136)
    img = np.random.rand(3, 64, 64)
    print(154, 139)
    img = conv_layer(img, filt, 1, zero_pad, stride)
    print(154, 140)
    weights = np.random.rand(len(img[0]))
    print(154, 141)
    biases = np.random.rand(len(img[0]))
    print(154, 142)
    img = BN_layer(img, 1, weights, biases)
    print(154, 143)
    img = reLU(img)
    print(154, 144)
    img = max_pool(img, 3, 3, 1, 2)
    print(154, 147)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    for i in range(2):
        print(156, 149)
        byPass = np.copy(img)
        print(156, 150)
        img = conv_layer(img, filt, 1, 3, 1)
        print(156, 151)
        img = BN_layer(img, 1, weights, biases)
        print(156, 152)
        img = reLU(img)
        print(156, 153)
        img = conv_layer(img, filt, 1, 3, 1)
        print(156, 154)
        img += byPass
        print(156, 155)
        img = reLU(img)
    print(157, 158)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(157, 159)
    byPass = np.copy(img)
    print(157, 160)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(157, 161)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(157, 162)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 163)
    img = BN_layer(img, 1, weights, biases)
    print(157, 164)
    img = reLU(img)
    print(157, 165)
    img = conv_layer(img, filt, 1, 3, 2)
    print(157, 166)
    img += byPass
    print(157, 167)
    img = reLU(img)
    print(157, 169)
    byPass = np.copy(img)
    print(157, 170)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 171)
    img = BN_layer(img, 1, weights, biases)
    print(157, 172)
    img = reLU(img)
    print(157, 173)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 174)
    img = BN_layer(img, 1, weights, biases)
    print(157, 175)
    img += byPass
    print(157, 176)
    img = reLU(img)
    print(157, 179)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(157, 180)
    byPass = np.copy(img)
    print(157, 181)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(157, 182)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(157, 183)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 184)
    img = BN_layer(img, 1, weights, biases)
    print(157, 185)
    img = reLU(img)
    print(157, 186)
    img = conv_layer(img, filt, 1, 3, 2)
    print(157, 187)
    img += byPass
    print(157, 188)
    img = reLU(img)
    print(157, 190)
    byPass = np.copy(img)
    print(157, 191)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 192)
    img = BN_layer(img, 1, weights, biases)
    print(157, 193)
    img = reLU(img)
    print(157, 194)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 195)
    img = BN_layer(img, 1, weights, biases)
    print(157, 196)
    img += byPass
    print(157, 197)
    img = reLU(img)
    print(157, 200)
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    print(157, 201)
    byPass = np.copy(img)
    print(157, 202)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    print(157, 203)
    byPass = BN_layer(byPass, 1, weights, biases)
    print(157, 204)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 205)
    img = BN_layer(img, 1, weights, biases)
    print(157, 206)
    img = reLU(img)
    print(157, 207)
    img = conv_layer(img, filt, 1, 3, 2)
    print(157, 208)
    img += byPass
    print(157, 209)
    img = reLU(img)
    print(157, 211)
    byPass = np.copy(img)
    print(157, 212)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 213)
    img = BN_layer(img, 1, weights, biases)
    print(157, 214)
    img = reLU(img)
    print(157, 215)
    img = conv_layer(img, filt, 1, 3, 1)
    print(157, 216)
    img = BN_layer(img, 1, weights, biases)
    print(157, 217)
    img += byPass
    print(157, 218)
    img = reLU(img)
    print(157, 219)
    img = avg_pool(img, len(img[0]), len(img[0][0]), 3, 1)
    print(157, 220)
    flat = flatten_layer(img)
    print(157, 222)
    weights = np.random.rand(len(img) * len(img[0]) * len(img[0][0]), 7)
    print(157, 223)
    w_0 = np.random.rand(7)
    print(157, 224)
    flat = fc_layer(flat, weights, w_0)
    print(157, 225)
    final = softmax(flat)
    return 0


if __name__ == '__main__':
    print(1, 229)
    main()
else:
    print(1, 229)
