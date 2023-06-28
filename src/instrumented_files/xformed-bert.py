import sys
import numpy as np
import math


def balance_random_3d(depth, l, wid):
    print(1, 4)
    print(3, 5)
    arr = np.random.rand(depth, l, wid)
    print(3, 6)
    neg = True
    for i in range(depth):
        for j in range(l):
            for k in range(wid):
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
    for i in range(l):
        for j in range(wid):
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
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            print(30, 28)
            dst[i][j] += src[i][j]
    return dst


def reLU(img):
    print(1, 31)
    for i in range(len(img)):
        for j in range(len(img[0])):
            print(38, 34)
            img[i][j] = max(img[i][j], 0)
    return img


def get_mean(row):
    print(1, 37)
    print(43, 38)
    sum_val = 0
    for i in range(len(row)):
        print(45, 40)
        sum_val += row[i]
    return sum_val / len(row)


def std_dev(row):
    print(1, 43)
    print(50, 44)
    result = 0
    for i in range(len(row)):
        print(52, 46)
        diff = row[i] - get_mean(row)
        print(52, 47)
        result += diff * diff
    return math.sqrt(result / len(row))


def BN_layer(arr, weights, biases):
    print(1, 50)
    for i in range(len(arr)):
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
        for j in range(len(arr[0])):
            print(62, 56)
            arr[i][j] = weights[i] * ((arr[i][j] - mean) / dev) + biases[i]
    return arr


def fc_layer(arr, W, W_0):
    print(1, 59)
    print(67, 60)
    result = np.zeros(len(W[0]))
    for i in range(len(W[0])):
        print(69, 62)
        sum_val = W_0[i]
        for j in range(len(arr)):
            print(72, 64)
            sum_val += arr[j] * W[j][i]
        print(73, 65)
        result[i] = sum_val
    return result


def softmax(arr):
    print(1, 68)
    print(77, 69)
    sum_val = 0
    for i in range(len(arr)):
        print(79, 70)
        sum_val += math.exp(arr[i])
    print(80, 71)
    result = np.zeros(len(arr))
    for i in range(len(arr)):
        print(82, 72)
        result[i] = math.exp(arr[i]) / sum_val
    return result


def concat(emb, head, tokens, d_k, cur):
    print(1, 75)
    for i in range(tokens):
        for j in range(d_k):
            print(90, 78)
            emb[i][j + head * d_k] = cur[i][j]
    return emb


def self_attn(head, tokens, d_k, Q, K, V):
    print(1, 81)
    print(95, 82)
    scores = np.zeros((tokens, tokens))
    for i in range(tokens):
        for j in range(tokens):
            for k in range(d_k):
                print(101, 86)
                scores[i][j] += Q[head][i][k] * K[head][j][k]
    for i in range(tokens):
        for j in range(tokens):
            print(105, 89)
            scores[i][j] /= math.sqrt(d_k)
        print(106, 91)
        scores = np.random.rand(tokens, tokens)
        print(106, 92)
        scores[i] = softmax(scores[i])
    print(104, 93)
    out = np.zeros((tokens, d_k))
    for i in range(tokens):
        for j in range(d_k):
            for k in range(tokens):
                print(112, 97)
                out[i][j] += scores[i][k] * V[head][k][j]
    return out


def main():
    print(1, 101)
    print(117, 102)
    d_model, heads, tokens, layers = 12, 12, 8, 12
    print(117, 103)
    d_k = d_model // heads
    print(117, 104)
    embeddings = np.random.rand(tokens, d_model)
    for i in range(tokens):
        for j in range(d_model):
            if j % 2 == 0:
                print(121, 107)
                print(123, 108)
                embeddings[i][j] += math.sin(i / math.pow(10000, 2 * j /
                    d_model))
            else:
                print(121, 107)
                print(125, 110)
                embeddings[i][j] += math.cos(i / math.pow(10000, 2 * j /
                    d_model))
    print(120, 111)
    W_Q = balance_random_3d(heads, d_model, d_k)
    print(120, 112)
    W_K = balance_random_3d(heads, d_model, d_k)
    print(120, 113)
    W_V = balance_random_3d(heads, d_model, d_k)
    print(120, 114)
    Q = np.zeros((heads, tokens, d_k))
    print(120, 115)
    K = np.zeros((heads, tokens, d_k))
    print(120, 116)
    V = np.zeros((heads, tokens, d_k))
    for i in range(heads):
        for j in range(tokens):
            for k in range(d_k):
                print(131, 120)
                sumQ, sumK, sumV = 0, 0, 0
                for a in range(d_model):
                    print(134, 122)
                    sumQ += embeddings[j][a] * W_Q[i][a][k]
                    print(134, 123)
                    sumK += embeddings[j][a] * W_K[i][a][k]
                    print(134, 124)
                    sumV += embeddings[j][a] * W_V[i][a][k]
                print(135, 125)
                Q[i][j][k] = sumQ
                print(135, 126)
                K[i][j][k] = sumK
                print(135, 127)
                V[i][j][k] = sumV
    for i in range(layers):
        print(136, 129)
        emb_cpy = np.copy(embeddings)
        print(136, 130)
        multi_head_out = np.zeros((tokens, d_model))
        for j in range(heads):
            print(139, 132)
            cur = self_attn(j, tokens, d_k, Q, K, V)
            print(139, 133)
            multi_head_out = concat(multi_head_out, j, tokens, d_k, cur)
        print(140, 134)
        W_attn = np.random.rand(d_model, d_model)
        for i in range(tokens):
            for j in range(d_model):
                print(144, 137)
                sum_val = 0
                for k in range(d_model):
                    print(147, 139)
                    sum_val += multi_head_out[i][k] * W_attn[k][j]
                print(148, 140)
                embeddings[i][j] = sum_val
        print(143, 141)
        embeddings = arr_add(embeddings, emb_cpy)
        print(143, 142)
        weights, biases = np.random.rand(d_model), np.random.rand(d_model)
        print(143, 143)
        embeddings = BN_layer(embeddings, weights, biases)
        print(143, 144)
        emb_cpy = np.copy(embeddings)
        print(143, 145)
        W = np.random.rand(d_model, d_model * 4)
        print(143, 146)
        W_0 = np.random.rand(d_model * 4)
        print(143, 147)
        emb_new = np.zeros((tokens, d_model * 4))
        for i in range(tokens):
            print(150, 149)
            emb_new[i] = fc_layer(embeddings[i], W, W_0)
        print(151, 150)
        embeddings = emb_new
        print(151, 151)
        embeddings = reLU(embeddings)
        print(151, 152)
        W = np.random.rand(d_model * 4, d_model)
        print(151, 153)
        W_0 = np.random.rand(d_model)
        print(151, 154)
        emb_new = np.zeros((tokens, d_model))
        for i in range(tokens):
            print(153, 156)
            emb_new[i] = fc_layer(embeddings[i], W, W_0)
        print(154, 157)
        embeddings = emb_new
        print(154, 158)
        embeddings = arr_add(embeddings, emb_cpy)
        print(154, 159)
        embeddings = BN_layer(embeddings, weights, biases)


if __name__ == '__main__':
    print(1, 161)
    main()
else:
    print(1, 161)
