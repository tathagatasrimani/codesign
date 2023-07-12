import sys
from instrument_lib import *
import numpy as np
import math


def balance_random_3d(depth, l, wid):
    print('enter scope 1')
    print(1, 4)
    depth__1 = depth
    l__1 = l
    wid__1 = wid
    print(3, 5)
    arr__1 = np.random.rand(depth__1, l__1, wid__1)
    print(3, 6)
    neg__1 = True
    for i__1 in range(depth__1):
        print('enter scope 2')
        for j__2 in range(l__1):
            print('enter scope 3')
            for k__3 in range(wid__1):
                print('enter scope 4')
                print('enter scope 5')
                if neg__1:
                    print(11, 11)
                    arr__1[i__1][j__2][k__3] *= -1
                print('exit scope 5')
                print(12, 12)
                neg__1 = not neg__1
                print('exit scope 4')
            print('exit scope 3')
        print('exit scope 2')
    print('exit scope 1')
    return arr__1
    print('exit scope 1')


def balance_random_2d(l, wid):
    print('enter scope 6')
    print(1, 15)
    l__6 = l
    wid__6 = wid
    print(16, 16)
    arr__6 = np.random.rand(l__6, wid__6)
    print(16, 17)
    neg__6 = True
    for i__6 in range(l__6):
        print('enter scope 7')
        for j__7 in range(wid__6):
            print('enter scope 8')
            print('enter scope 9')
            if neg__6:
                print(22, 21)
                arr__6[i__6][j__7] *= -1
            print('exit scope 9')
            print(23, 22)
            neg__6 = not neg__6
            print('exit scope 8')
        print('exit scope 7')
    print('exit scope 6')
    return arr__6
    print('exit scope 6')


def arr_add(dst, src):
    print('enter scope 10')
    print(1, 25)
    dst__10 = dst
    src__10 = src
    for i__10 in range(len(dst__10)):
        print('enter scope 11')
        for j__11 in range(len(dst__10[0])):
            print('enter scope 12')
            print(30, 28)
            dst__10[i__10][j__11] += src__10[i__10][j__11]
            print('exit scope 12')
        print('exit scope 11')
    print('exit scope 10')
    return dst__10
    print('exit scope 10')


def reLU(img):
    print('enter scope 13')
    print(1, 31)
    img__13 = img
    for i__13 in range(len(img__13)):
        print('enter scope 14')
        for j__14 in range(len(img__13[0])):
            print('enter scope 15')
            print(38, 34)
            img__13[i__13][j__14] = max(img__13[i__13][j__14], 0)
            print('exit scope 15')
        print('exit scope 14')
    print('exit scope 13')
    return img__13
    print('exit scope 13')


def get_mean(row):
    print('enter scope 16')
    print(1, 37)
    row__16 = row
    print(43, 38)
    sum_val__16 = 0
    for i__16 in range(len(row__16)):
        print('enter scope 17')
        print(45, 40)
        sum_val__16 += row__16[i__16]
        print('exit scope 17')
    print('exit scope 16')
    return sum_val__16 / len(row__16)
    print('exit scope 16')


def std_dev(row):
    print('enter scope 18')
    print(1, 43)
    row__18 = row
    print(50, 44)
    result__18 = 0
    for i__18 in range(len(row__18)):
        print('enter scope 19')
        print(52, 46)
        diff__19 = row__18[i__18] - get_mean(row__18)
        print(52, 47)
        result__18 += diff__19 * diff__19
        print('exit scope 19')
    print('exit scope 18')
    return math.sqrt(result__18 / len(row__18))
    print('exit scope 18')


def BN_layer(arr, weights, biases):
    print('enter scope 20')
    print(1, 50)
    arr__20 = arr
    weights__20 = weights
    biases__20 = biases
    for i__20 in range(len(arr__20)):
        print('enter scope 21')
        print(58, 52)
        dev__21 = std_dev(arr__20[i__20])
        print(58, 53)
        mean__21 = get_mean(arr__20[i__20])
        print('enter scope 22')
        if dev__21 == 0:
            print(60, 54)
            dev__21 = 1
        print('exit scope 22')
        for j__21 in range(len(arr__20[0])):
            print('enter scope 23')
            print(62, 56)
            arr__20[i__20][j__21] = weights__20[i__20] * ((arr__20[i__20][
                j__21] - mean__21) / dev__21) + biases__20[i__20]
            print('exit scope 23')
        print('exit scope 21')
    print('exit scope 20')
    return arr__20
    print('exit scope 20')


def fc_layer(arr, W, W_0):
    print('enter scope 24')
    print(1, 59)
    arr__24 = arr
    W__24 = W
    W_0__24 = W_0
    print(67, 60)
    result__24 = np.zeros(len(W__24[0]))
    for i__24 in range(len(W__24[0])):
        print('enter scope 25')
        print(69, 62)
        sum_val__25 = W_0__24[i__24]
        for j__25 in range(len(arr__24)):
            print('enter scope 26')
            print(72, 64)
            sum_val__25 += arr__24[j__25] * W__24[j__25][i__24]
            print('exit scope 26')
        print(73, 65)
        result__24[i__24] = sum_val__25
        print('exit scope 25')
    print('exit scope 24')
    return result__24
    print('exit scope 24')


def softmax(arr):
    print('enter scope 27')
    print(1, 68)
    arr__27 = arr
    print(77, 69)
    sum_val__27 = 0
    for i__27 in range(len(arr__27)):
        print('enter scope 28')
        print(79, 70)
        sum_val__27 += math.exp(arr__27[i__27])
        print('exit scope 28')
    print(80, 71)
    result__27 = np.zeros(len(arr__27))
    for i__27 in range(len(arr__27)):
        print('enter scope 29')
        print(82, 72)
        result__27[i__27] = math.exp(arr__27[i__27]) / sum_val__27
        print('exit scope 29')
    print('exit scope 27')
    return result__27
    print('exit scope 27')


def concat(emb, head, tokens, d_k, cur):
    print('enter scope 30')
    print(1, 75)
    emb__30 = emb
    head__30 = head
    tokens__30 = tokens
    d_k__30 = d_k
    cur__30 = cur
    for i__30 in range(tokens__30):
        print('enter scope 31')
        for j__31 in range(d_k__30):
            print('enter scope 32')
            print(90, 78)
            emb__30[i__30][j__31 + head__30 * d_k__30] = cur__30[i__30][j__31]
            print('exit scope 32')
        print('exit scope 31')
    print('exit scope 30')
    return emb__30
    print('exit scope 30')


def self_attn(head, tokens, d_k, Q, K, V):
    print('enter scope 33')
    print(1, 81)
    head__33 = head
    tokens__33 = tokens
    d_k__33 = d_k
    Q__33 = Q
    K__33 = K
    V__33 = V
    print(95, 82)
    scores__33 = np.zeros((tokens__33, tokens__33))
    for i__33 in range(tokens__33):
        print('enter scope 34')
        for j__34 in range(tokens__33):
            print('enter scope 35')
            for k__35 in range(d_k__33):
                print('enter scope 36')
                print(101, 86)
                scores__33[i__33][j__34] += Q__33[head__33][i__33][k__35
                    ] * K__33[head__33][j__34][k__35]
                print('exit scope 36')
            print('exit scope 35')
        print('exit scope 34')
    for i__33 in range(tokens__33):
        print('enter scope 37')
        for j__37 in range(tokens__33):
            print('enter scope 38')
            print(105, 89)
            scores__33[i__33][j__37] /= math.sqrt(d_k__33)
            print('exit scope 38')
        print(106, 91)
        scores__33 = np.random.rand(tokens__33, tokens__33)
        print(106, 92)
        scores__33[i__33] = softmax(scores__33[i__33])
        print('exit scope 37')
    print(104, 93)
    out__33 = np.zeros((tokens__33, d_k__33))
    for i__33 in range(tokens__33):
        print('enter scope 39')
        for j__39 in range(d_k__33):
            print('enter scope 40')
            for k__40 in range(tokens__33):
                print('enter scope 41')
                print(112, 97)
                out__33[i__33][j__39] += scores__33[i__33][k__40] * V__33[
                    head__33][k__40][j__39]
                print('exit scope 41')
            print('exit scope 40')
        print('exit scope 39')
    print('exit scope 33')
    return out__33
    print('exit scope 33')


def main():
    print('enter scope 42')
    print(1, 101)
    print(117, 102)
    d_model__42, heads__42, tokens__42, layers__42 = 12, 12, 8, 12
    print(117, 103)
    d_k__42 = d_model__42 // heads__42
    print(117, 104)
    embeddings__42 = np.random.rand(tokens__42, d_model__42)
    for i__42 in range(tokens__42):
        print('enter scope 43')
        for j__43 in range(d_model__42):
            print('enter scope 44')
            print('enter scope 45')
            if j__43 % 2 == 0:
                print(123, 108)
                embeddings__42[i__42][j__43] += math.sin(i__42 / math.pow(
                    10000, 2 * j__43 / d_model__42))
            else:
                print(125, 110)
                embeddings__42[i__42][j__43] += math.cos(i__42 / math.pow(
                    10000, 2 * j__43 / d_model__42))
            print('exit scope 45')
            print('exit scope 44')
        print('exit scope 43')
    print(120, 111)
    W_Q__42 = balance_random_3d(heads__42, d_model__42, d_k__42)
    print(120, 112)
    W_K__42 = balance_random_3d(heads__42, d_model__42, d_k__42)
    print(120, 113)
    W_V__42 = balance_random_3d(heads__42, d_model__42, d_k__42)
    print(120, 114)
    Q__42 = np.zeros((heads__42, tokens__42, d_k__42))
    print(120, 115)
    K__42 = np.zeros((heads__42, tokens__42, d_k__42))
    print(120, 116)
    V__42 = np.zeros((heads__42, tokens__42, d_k__42))
    for i__42 in range(heads__42):
        print('enter scope 46')
        for j__46 in range(tokens__42):
            print('enter scope 47')
            for k__47 in range(d_k__42):
                print('enter scope 48')
                print(131, 120)
                sumQ__48, sumK__48, sumV__48 = 0, 0, 0
                for a__48 in range(d_model__42):
                    print('enter scope 49')
                    print(134, 122)
                    sumQ__48 += embeddings__42[j__46][a__48] * W_Q__42[i__42][
                        a__48][k__47]
                    print(134, 123)
                    sumK__48 += embeddings__42[j__46][a__48] * W_K__42[i__42][
                        a__48][k__47]
                    print(134, 124)
                    sumV__48 += embeddings__42[j__46][a__48] * W_V__42[i__42][
                        a__48][k__47]
                    print('exit scope 49')
                print(135, 125)
                Q__42[i__42][j__46][k__47] = sumQ__48
                print(135, 126)
                K__42[i__42][j__46][k__47] = sumK__48
                print(135, 127)
                V__42[i__42][j__46][k__47] = sumV__48
                print('exit scope 48')
            print('exit scope 47')
        print('exit scope 46')
    for i__42 in range(layers__42):
        print('enter scope 50')
        print(136, 129)
        emb_cpy__50 = np.copy(embeddings__42)
        print(136, 130)
        multi_head_out__50 = np.zeros((tokens__42, d_model__42))
        for j__50 in range(heads__42):
            print('enter scope 51')
            print(139, 132)
            cur__51 = self_attn(j__50, tokens__42, d_k__42, Q__42, K__42, V__42
                )
            print(139, 133)
            multi_head_out__50 = concat(multi_head_out__50, j__50,
                tokens__42, d_k__42, cur__51)
            print('exit scope 51')
        print(140, 134)
        W_attn__50 = np.random.rand(d_model__42, d_model__42)
        for i__42 in range(tokens__42):
            print('enter scope 52')
            for j__50 in range(d_model__42):
                print('enter scope 53')
                print(144, 137)
                sum_val__53 = 0
                for k__53 in range(d_model__42):
                    print('enter scope 54')
                    print(147, 139)
                    sum_val__53 += multi_head_out__50[i__42][k__53
                        ] * W_attn__50[k__53][j__50]
                    print('exit scope 54')
                print(148, 140)
                embeddings__42[i__42][j__50] = sum_val__53
                print('exit scope 53')
            print('exit scope 52')
        print(143, 141)
        embeddings__42 = arr_add(embeddings__42, emb_cpy__50)
        print(143, 142)
        weights__50, biases__50 = np.random.rand(d_model__42), np.random.rand(
            d_model__42)
        print(143, 143)
        embeddings__42 = BN_layer(embeddings__42, weights__50, biases__50)
        print(143, 144)
        emb_cpy__50 = np.copy(embeddings__42)
        print(143, 145)
        W__50 = np.random.rand(d_model__42, d_model__42 * 4)
        print(143, 146)
        W_0__50 = np.random.rand(d_model__42 * 4)
        print(143, 147)
        emb_new__50 = np.zeros((tokens__42, d_model__42 * 4))
        for i__42 in range(tokens__42):
            print('enter scope 55')
            print(150, 149)
            emb_new__50[i__42] = fc_layer(embeddings__42[i__42], W__50, W_0__50
                )
            print('exit scope 55')
        print(151, 150)
        embeddings__42 = emb_new__50
        print(151, 151)
        embeddings__42 = reLU(embeddings__42)
        print(151, 152)
        W__50 = np.random.rand(d_model__42 * 4, d_model__42)
        print(151, 153)
        W_0__50 = np.random.rand(d_model__42)
        print(151, 154)
        emb_new__50 = np.zeros((tokens__42, d_model__42))
        for i__42 in range(tokens__42):
            print('enter scope 56')
            print(153, 156)
            emb_new__50[i__42] = fc_layer(embeddings__42[i__42], W__50, W_0__50
                )
            print('exit scope 56')
        print(154, 157)
        embeddings__42 = emb_new__50
        print(154, 158)
        embeddings__42 = arr_add(embeddings__42, emb_cpy__50)
        print(154, 159)
        embeddings__42 = BN_layer(embeddings__42, weights__50, biases__50)
        print('exit scope 50')
    print('exit scope 42')


print('enter scope 57')
if __name__ == '__main__':
    main()
print('exit scope 57')
