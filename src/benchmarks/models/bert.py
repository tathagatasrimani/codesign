import numpy as np
import math
from loop import loop

def balance_random_3d(depth, l, wid):
    arr = np.random.rand(depth, l, wid)
    neg = True
    for i in range(depth):
        for j in range(l):
            for k in range(wid):
                if neg:
                    arr[i][j][k] *= -1
                neg = not neg
    return arr

def balance_random_2d(l, wid):
    arr = np.random.rand(l, wid)
    neg = True
    for i in range(l):
        for j in range(wid):
            if neg:
                arr[i][j] *= -1
            neg = not neg
    return arr

def arr_add(dst, src):
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            dst[i][j] += src[i][j]
    return dst

def reLU(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = max(img[i][j], 0)
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

def BN_layer(arr, weights, biases):
    for i in range(len(arr)):
        dev = std_dev(arr[i])
        mean = get_mean(arr[i])
        if dev == 0: dev = 1
        for j in range(len(arr[0])):
            arr[i][j] = weights[i] * ((arr[i][j] - mean) / dev) + biases[i]
    return arr

def fc_layer(arr, W, W_0):
    result = np.zeros(len(W[0]))
    for i in range(len(W[0])):
        sum_val = W_0[i]
        for j in range(len(arr)):
            sum_val += arr[j] * W[j][i]
        result[i] = sum_val
    return result

def softmax(arr):
    sum_val = 0
    for i in range(len(arr)): sum_val += math.exp(arr[i])
    result = np.zeros(len(arr))
    for i in range(len(arr)): result[i] = math.exp(arr[i]) / sum_val
    return result

def concat(emb, head, tokens, d_k, cur):
    for i in range(tokens):
        for j in range(d_k):
            emb[i][j + head * d_k] = cur[i][j]
    return emb

def self_attn(head, tokens, d_k, Q, K, V):
    scores = np.zeros((tokens, tokens))
    for i in range(tokens):
        for j in range(tokens):
            for k in range(d_k):
                scores[i][j] += Q[head][i][k] * K[head][j][k]
    for i in range(tokens):
        for j in range(tokens):
            scores[i][j] /= math.sqrt(d_k)
        # avoid overflow
        scores = np.random.rand(tokens, tokens)
        scores[i] = softmax(scores[i])
    out = np.zeros((tokens, d_k))
    for i in range(tokens):
        for j in range(d_k):
            for k in range(tokens):
                out[i][j] += scores[i][k] * V[head][k][j]
    return out


def main():
    d_model, heads, tokens, layers = 20, 12, 8, 12
    d_k = d_model // heads
    embeddings = np.random.rand(tokens, d_model)
    for i in range(tokens):
        for j in range(d_model):
            if j % 2 == 0:
                embeddings[i][j] += math.sin(i / math.pow(10000, (2 * j / d_model)))
            else:
                embeddings[i][j] += math.cos(i / math.pow(10000, (2 * j / d_model)))
    W_Q = balance_random_3d(heads, d_model, d_k)
    W_K = balance_random_3d(heads, d_model, d_k)
    W_V = balance_random_3d(heads, d_model, d_k)
    Q = np.zeros((heads, tokens, d_k))
    K = np.zeros((heads, tokens, d_k))
    V = np.zeros((heads, tokens, d_k))
    for i in range(heads):
        for j in range(tokens):
            for k in range(d_k):
                sumQ, sumK, sumV = 0, 0, 0
                for a in range(d_model):
                    sumQ += embeddings[j][a] * W_Q[i][a][k]
                    sumK += embeddings[j][a] * W_K[i][a][k]
                    sumV += embeddings[j][a] * W_V[i][a][k]
                Q[i][j][k] = sumQ
                K[i][j][k] = sumK
                V[i][j][k] = sumV
    for i in range(layers):
        emb_cpy = np.copy(embeddings)
        multi_head_out = np.zeros((tokens, d_model))
        for j in range(heads):
            cur = self_attn(j, tokens, d_k, Q, K, V)
            multi_head_out = concat(multi_head_out, j, tokens, d_k, cur)
        W_attn = np.random.rand(d_model, d_model)
        for i in range(tokens):
            for j in range(d_model):
                sum_val = 0
                for k in range(d_model):
                    sum_val += multi_head_out[i][k] * W_attn[k][j]
                embeddings[i][j] = sum_val
        embeddings = arr_add(embeddings, emb_cpy)
        weights, biases = np.random.rand(d_model), np.random.rand(d_model)
        embeddings = BN_layer(embeddings, weights, biases)
        emb_cpy = np.copy(embeddings)
        W = np.random.rand(d_model, d_model * 4)
        W_0 = np.random.rand(d_model * 4)
        emb_new = np.zeros((tokens, d_model * 4))
        for i in range(tokens):
            emb_new[i] = fc_layer(embeddings[i], W, W_0)
        embeddings = emb_new
        embeddings = reLU(embeddings)
        W = np.random.rand(d_model * 4, d_model)
        W_0 = np.random.rand(d_model)
        emb_new = np.zeros((tokens, d_model))
        for i in range(tokens):
            emb_new[i] = fc_layer(embeddings[i], W, W_0)
        embeddings = emb_new
        embeddings = arr_add(embeddings, emb_cpy)
        embeddings = BN_layer(embeddings, weights, biases)

if __name__ == "__main__":
    loop.start_unroll
    main()