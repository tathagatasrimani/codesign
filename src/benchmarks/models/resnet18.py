import math
import numpy as np
from loop import loop

def zero_pad_arr(img, zero_pad):
    len_new = len(img[0]) + 2 * zero_pad
    wid_new = len(img[0][0]) + 2 * zero_pad
    new_img = np.zeros((len(img), len_new, wid_new))
    for i in range(len(img)):
        for j in range(len_new):
            make_zero = j < zero_pad or j >= len_new - zero_pad
            for k in range(wid_new):
                if k < zero_pad or k >= wid_new - zero_pad or make_zero:
                    new_img[i][j][k] = 0
                else:
                    new_img[i][j][k] = img[i][j-zero_pad][k-zero_pad]
    return new_img


def conv_layer(img, filt, numFilt, zero_pad, stride):
    f_len = int((len(img[0]) - len(filt[0]) + 2 * zero_pad) / stride + 1)
    f_wid = int((len(img[0][0]) - len(filt[0][0]) + 2 * zero_pad) / stride + 1)
    biases = np.random.rand(f_len)
    img_new = zero_pad_arr(img, zero_pad)
    f_new = np.zeros((numFilt, f_len, f_wid))
    for i in range(numFilt):
        for j in range(f_len):
            for k in range(f_wid):
                for l in range(len(filt)):
                    for c in range(len(filt[0])):
                        for d in range(len(filt[0][0])):
                            f_new[i][j][k] += img_new[l][j * stride + c][k * stride + d] * filt[l][c][d]
    for i in range(numFilt):
        for j in range(f_len):
            for k in range(f_wid):
                f_new[i][j][k] += biases[j]
    return f_new

def max_pool(input, l, w, zero_pad, stride):
    if zero_pad > 0:
        input = zero_pad_arr(input, zero_pad)
    res_l = int((len(input[0]) - l) / stride + 1)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    result = np.zeros((len(input), res_l, res_w)) 
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                for c in range(l):
                    for d in range(w):
                        result[i][j][k] = max(input[i][j][k], result[i][j][k], input[i][j * stride + c][k * stride + d])
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                result[i][j][k] /= l * w
    return result

def avg_pool(input, l, w, zero_pad, stride):
    if zero_pad > 0:
        input = zero_pad_arr(input, zero_pad)
    res_l = int((len(input[0]) - l) / stride + 1)
    res_w = int((len(input[0][0]) - w) / stride + 1)
    result = np.zeros((len(input), res_l, res_w))
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                for c in range(l):
                    for d in range(w):
                        result[i][j][k] += input[i][j * stride + c][k * stride + d]
    for i in range(len(input)):
        for j in range(res_l):
            for k in range(res_w):
                result[i][j][k] /= l * w
    return result

def reLU(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                img[i][j][k] = max(img[i][j][k], 0)
    return img

def get_mean(row):
    sum_val = 0
    for i in range(len(row)):
        sum_val += row[i]
    return sum_val / len(row)

def std_dev(row):
    result = 0
    for i in range(len(row)):
        diff = row[i] - get_mean(row)
        result += diff * diff
    return math.sqrt(result / len(row))

def BN_layer(img, channels, weights, biases):
    for i in range(channels):
        for j in range(len(img[0])):
            dev = std_dev(img[i][j][:])
            mean = get_mean(img[i][j][:])
            if dev == 0.0: dev = 1.0
            for k in range(len(img[0][0])):
                img[i][j][k] = weights[j] * ((img[i][j][k] - mean) / dev) + biases[j]
    return img

def flatten_layer(img):
    result = np.zeros(len(img) * len(img[0]) * len(img[0][0]))
    index = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                result[index] = img[i][j][k]
                index += 1
    return result

def fc_layer(arr, W, W_0):
    result = np.zeros(len(W[0]))
    for i in range(len(W[0])):
        sum_val = W_0[i]
        for j in range(len(arr)):
            sum_val += arr[j] * W[j][i]
        result[i] = sum_val
    return result

def softmax(img):
    sum_val = 0
    for i in range(len(img)):
        sum_val += math.exp(img[i])
    result = np.zeros(len(img))
    for i in range(len(img)):
        result[i] = math.exp(img[i]) / sum_val
    return result

def main():
    zero_pad = 3
    stride = 2
    filt = np.random.rand(3, 7, 7)
    img = np.random.rand(3, 24, 24)

    # Block 1
    img = conv_layer(img, filt, 1, zero_pad, stride)
    weights = np.random.rand(len(img[0]))
    biases = np.random.rand(len(img[0]))
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = max_pool(img, 3, 3, 1, 2)

    # Block 2
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    for i in range(2):
        byPass = np.copy(img)
        img = conv_layer(img, filt, 1, 3, 1)
        img = BN_layer(img, 1, weights, biases)
        img = reLU(img)
        img = conv_layer(img, filt, 1, 3, 1)
        img += byPass
        img = reLU(img)

    # Block 3
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    byPass = np.copy(img)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    byPass = BN_layer(byPass, 1, weights, biases)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = conv_layer(img, filt, 1, 3, 2)
    img += byPass
    img = reLU(img)

    byPass = np.copy(img)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img += byPass
    img = reLU(img)

    # Block 4
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    byPass = np.copy(img)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    byPass = BN_layer(byPass, 1, weights, biases)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = conv_layer(img, filt, 1, 3, 2)
    img += byPass
    img = reLU(img)

    byPass = np.copy(img)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img += byPass
    img = reLU(img)

    # Block 5
    filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
    byPass = np.copy(img)
    byPass = conv_layer(byPass, filt, 1, 3, 2)
    byPass = BN_layer(byPass, 1, weights, biases)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = conv_layer(img, filt, 1, 3, 2)
    img += byPass
    img = reLU(img)

    byPass = np.copy(img)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img = reLU(img)
    img = conv_layer(img, filt, 1, 3, 1)
    img = BN_layer(img, 1, weights, biases)
    img += byPass
    img = reLU(img)
    img = avg_pool(img, len(img[0]), len(img[0][0]), 3, 1)
    flat = flatten_layer(img)

    weights = np.random.rand(len(img) * len(img[0]) * len(img[0][0]), 7)
    w_0 = np.random.rand(7)
    flat = fc_layer(flat, weights, w_0)
    final = softmax(flat)

    return 0

if __name__ == "__main__":
    loop.start_unroll
    main()