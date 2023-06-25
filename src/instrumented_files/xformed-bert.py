import sys
import numpy as np
import math


def balance_random_3d(depth, l, wid):
    print(1, 4)
    print(3, 5)
    arr = np.random.rand(depth, l, wid)
    print(3, 6)
    neg = True
    print(4, 7)
    for i in range(depth):
        print(4, 7)
        print(5, 8)
        for j in range(l):
            print(5, 8)
            print(7, 9)
            for k in range(wid):
                print(7, 9)
                if neg:
                    print(9, 10)
                    print(11, 11)
                    arr[i][j][k] *= -1
                else:
                    print(9, 10)
                print(12, 12)
                neg = not neg
    return arr


def balance_random_2d(l, wid):
    print(1, 15)
    print(16, 16)
    arr = np.random.rand(l, wid)
    print(16, 17)
    neg = True
    print(17, 18)
    for i in range(l):
        print(17, 18)
        print(18, 19)
        for j in range(wid):
            print(18, 19)
            if neg:
                print(20, 20)
                print(22, 21)
                arr[i][j] *= -1
            else:
                print(20, 20)
            print(23, 22)
            neg = not neg
    return arr


def arr_add(dst, src):
    print(1, 25)
    print(27, 26)
    for i in range(len(dst)):
        print(27, 26)
        print(28, 27)
        for j in range(len(dst[0])):
            print(28, 27)
            print(30, 28)
            dst[i][j] += src[i][j]
    return dst


def reLU(img):
    print(1, 31)
    print(35, 32)
    for i in range(len(img)):
        print(35, 32)
        print(36, 33)
        for j in range(len(img[0])):
            print(36, 33)
            print(38, 34)
            img[i][j] = max(img[i][j], 0)
    return img


def get_mean(row):
    print(1, 37)
    print(43, 38)
    sum_val = 0
    print(44, 39)
    for i in range(len(row)):
        print(44, 39)
        print(45, 40)
        sum_val += row[i]
    return sum_val / len(row)


def std_dev(row):
    print(1, 43)
    print(50, 44)
    result = 0
    print(51, 45)
    for i in range(len(row)):
        print(51, 45)
        print(52, 46)
        diff = row[i] - get_mean(row)
        print(52, 47)
        result += diff * diff
    return math.sqrt(result / len(row))


def BN_layer(arr, weights, biases):
    print(1, 50)
    print(57, 51)
    for i in range(len(arr)):
        print(57, 51)
        print(58, 52)
        dev = std_dev(arr[i])
        print(58, 53)
        mean = get_mean(arr[i])
        if dev == 0:
            print(60, 54)
            print(60, 54)
            dev = 1
        else:
            print(60, 54)
        print(61, 55)
        for j in range(len(arr[0])):
            print(61, 55)
            print(62, 56)
            arr[i][j] = weights[i] * ((arr[i][j] - mean) / dev) + biases[i]
    return arr


def fc_layer(arr, W, W_0):
    print(1, 59)
    print(67, 60)
    result = np.zeros(len(W[0]))
    print(68, 61)
    for i in range(len(W[0])):
        print(68, 61)
        print(69, 62)
        sum_val = W_0[i]
        print(71, 63)
        for j in range(len(arr)):
            print(71, 63)
            print(72, 64)
            sum_val += arr[j] * W[j][i]
        print(73, 65)
        result[i] = sum_val
    return result


def softmax(arr):
    print(1, 68)
    print(77, 69)
    sum_val = 0
    print(79, 70)
    for i in range(len(arr)):
        print(79, 70)
        print(79, 70)
        sum_val += math.exp(arr[i])
    print(80, 71)
    result = np.zeros(len(arr))
    print(82, 72)
    for i in range(len(arr)):
        print(82, 72)
        print(82, 72)
        result[i] = math.exp(arr[i]) / sum_val
    return result


def concat(emb, head, tokens, d_k, cur):
    print(1, 75)
    print(87, 76)
    for i in range(tokens):
        print(87, 76)
        print(88, 77)
        for j in range(d_k):
            print(88, 77)
            print(90, 78)
            emb[i][j + head * d_k] = cur[i][j]
    return emb


def self_attn(head, tokens, d_k, Q, K, V):
    print(1, 81)
    print(95, 82)
    scores = np.zeros((tokens, tokens))
    print(96, 83)
    for i in range(tokens):
        print(96, 83)
        print(97, 84)
        for j in range(tokens):
            print(97, 84)
            print(99, 85)
            sum_val = 0
            print(101, 86)
            for k in range(d_k):
                print(101, 86)
                print(102, 87)
                sum_val += Q[head][i][k] * K[head][j][k]
            print(103, 88)
            val = math.sqrt(d_k)
            if val == 0:
                print(104, 89)
                print(104, 89)
                val = 1
            else:
                print(104, 89)
            print(105, 90)
            scores[i][j] = sum_val / val
        print(100, 92)
        scores = np.random.rand(tokens, tokens)
        print(100, 93)
        scores[i] = softmax(scores[i])
    print(98, 94)
    out = np.zeros((tokens, d_k))
    print(106, 95)
    for i in range(tokens):
        print(106, 95)
        print(107, 96)
        for j in range(d_k):
            print(107, 96)
            print(109, 97)
            sum_val = 0
            print(111, 98)
            for k in range(tokens):
                print(111, 98)
                print(112, 99)
                sum_val += scores[i][k] * V[head][k][j]
            print(113, 100)
            out[i][j] = sum_val
    return out


def main():
    print(1, 104)
    print(117, 105)
    d_model, heads, tokens, layers = 12, 12, 8, 12
    print(117, 106)
    d_k = d_model // heads
    print(117, 107)
    embeddings = np.random.rand(tokens, d_model)
    print(118, 108)
    for i in range(tokens):
        print(118, 108)
        print(119, 109)
        for j in range(d_model):
            print(119, 109)
            if j % 2 == 0:
                print(121, 110)
                print(123, 111)
                embeddings[i][j] += math.sin(i / math.pow(10000, 2 * j /
                    d_model))
            else:
                print(121, 110)
                print(125, 113)
                embeddings[i][j] += math.cos(i / math.pow(10000, 2 * j /
                    d_model))
    print(120, 114)
    W_Q = balance_random_3d(heads, d_model, d_k)
    print(120, 115)
    W_K = balance_random_3d(heads, d_model, d_k)
    print(120, 116)
    W_V = balance_random_3d(heads, d_model, d_k)
    print(120, 117)
    Q = np.zeros((heads, tokens, d_k))
    print(120, 118)
    K = np.zeros((heads, tokens, d_k))
    print(120, 119)
    V = np.zeros((heads, tokens, d_k))
    print(126, 120)
    for i in range(heads):
        print(126, 120)
        print(127, 121)
        for j in range(tokens):
            print(127, 121)
            print(129, 122)
            for k in range(d_k):
                print(129, 122)
                print(131, 123)
                sumQ, sumK, sumV = 0, 0, 0
                print(133, 124)
                for a in range(d_model):
                    print(133, 124)
                    print(134, 125)
                    sumQ += embeddings[j][a] * W_Q[i][a][k]
                    print(134, 126)
                    sumK += embeddings[j][a] * W_K[i][a][k]
                    print(134, 127)
                    sumV += embeddings[j][a] * W_V[i][a][k]
                print(135, 128)
                Q[i][j][k] = sumQ
                print(135, 129)
                K[i][j][k] = sumK
                print(135, 130)
                V[i][j][k] = sumV
    print(128, 131)
    for i in range(layers):
        print(128, 131)
        print(136, 132)
        emb_cpy = np.copy(embeddings)
        print(136, 133)
        multi_head_out = np.zeros((tokens, d_model))
        print(138, 134)
        for j in range(heads):
            print(138, 134)
            print(139, 135)
            cur = self_attn(j, tokens, d_k, Q, K, V)
            print(139, 136)
            multi_head_out = concat(multi_head_out, j, tokens, d_k, cur)
        print(140, 137)
        W_attn = np.random.rand(d_model, d_model)
        print(141, 138)
        for i in range(tokens):
            print(141, 138)
            print(142, 139)
            for j in range(d_model):
                print(142, 139)
                print(144, 140)
                sum_val = 0
                print(146, 141)
                for k in range(d_model):
                    print(146, 141)
                    print(147, 142)
                    sum_val += multi_head_out[i][k] * W_attn[k][j]
                print(148, 143)
                embeddings[i][j] = sum_val
        print(143, 144)
        embeddings = arr_add(embeddings, emb_cpy)
        print(143, 145)
        weights, biases = np.random.rand(d_model), np.random.rand(d_model)
        print(143, 146)
        embeddings = BN_layer(embeddings, weights, biases)
        print(143, 147)
        emb_cpy = np.copy(embeddings)
        print(143, 148)
        W = np.random.rand(d_model, d_model * 4)
        print(143, 149)
        W_0 = np.random.rand(d_model * 4)
        print(143, 150)
        emb_new = np.zeros((tokens, d_model * 4))
        print(149, 151)
        for i in range(tokens):
            print(149, 151)
            print(150, 152)
            emb_new[i] = fc_layer(embeddings[i], W, W_0)
        print(151, 153)
        embeddings = emb_new
        print(151, 154)
        embeddings = reLU(embeddings)
        print(151, 155)
        W = np.random.rand(d_model * 4, d_model)
        print(151, 156)
        W_0 = np.random.rand(d_model)
        print(151, 157)
        emb_new = np.zeros((tokens, d_model))
        print(152, 158)
        for i in range(tokens):
            print(152, 158)
            print(153, 159)
            emb_new[i] = fc_layer(embeddings[i], W, W_0)
        print(154, 160)
        embeddings = emb_new
        print(154, 161)
        embeddings = arr_add(embeddings, emb_cpy)
        print(154, 162)
        embeddings = BN_layer(embeddings, weights, biases)


if __name__ == '__main__':
    print(1, 164)
    main()
else:
    print(1, 164)
