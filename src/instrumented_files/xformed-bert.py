import sys
from instrument_lib import *
import sys
from instrument_lib import *
import numpy as np
import math


def balance_random_3d(depth, l, wid):
    print('enter scope 1')
    print(1, 4)
    depth__1 = instrument_read(depth, 'depth')
    write_instrument_read(depth__1, 'depth__1')
    print('malloc', sys.getsizeof(depth__1), 'depth__1')
    l__1 = instrument_read(l, 'l')
    write_instrument_read(l__1, 'l__1')
    print('malloc', sys.getsizeof(l__1), 'l__1')
    wid__1 = instrument_read(wid, 'wid')
    write_instrument_read(wid__1, 'wid__1')
    print('malloc', sys.getsizeof(wid__1), 'wid__1')
    print(3, 5)
    arr__1 = instrument_read(np, 'np').random.rand(instrument_read(depth__1,
        'depth__1'), instrument_read(l__1, 'l__1'), instrument_read(wid__1,
        'wid__1'))
    write_instrument_read(arr__1, 'arr__1')
    print('malloc', sys.getsizeof(arr__1), 'arr__1')
    print(3, 6)
    neg__1 = True
    write_instrument_read(neg__1, 'neg__1')
    print('malloc', sys.getsizeof(neg__1), 'neg__1')
    for i__1 in range(instrument_read(depth__1, 'depth__1')):
        print('enter scope 2')
        for j__2 in range(instrument_read(l__1, 'l__1')):
            print('enter scope 3')
            for k__3 in range(instrument_read(wid__1, 'wid__1')):
                print('enter scope 4')
                print('enter scope 5')
                if instrument_read(neg__1, 'neg__1'):
                    print(11, 11)
                    arr__1[instrument_read(i__1, 'i__1')][instrument_read(
                        j__2, 'j__2')][instrument_read(k__3, 'k__3')] *= -1
                    write_instrument_read_sub(arr__1[instrument_read(
                        instrument_read(i__1, 'i__1'), 'i__1')][
                        instrument_read(instrument_read(j__2, 'j__2'),
                        'j__2')],
                        "arr__1[instrument_read(i__1, 'i__1')][instrument_read(j__2, 'j__2')]"
                        , instrument_read(instrument_read(k__3, 'k__3'),
                        'k__3'), None, None, False)
                print('exit scope 5')
                print(12, 12)
                neg__1 = not instrument_read(neg__1, 'neg__1')
                write_instrument_read(neg__1, 'neg__1')
                print('malloc', sys.getsizeof(neg__1), 'neg__1')
                print('exit scope 4')
            print('exit scope 3')
        print('exit scope 2')
    print('exit scope 1')
    return instrument_read(arr__1, 'arr__1')
    print('exit scope 1')


def balance_random_2d(l, wid):
    print('enter scope 6')
    print(1, 15)
    l__6 = instrument_read(l, 'l')
    write_instrument_read(l__6, 'l__6')
    print('malloc', sys.getsizeof(l__6), 'l__6')
    wid__6 = instrument_read(wid, 'wid')
    write_instrument_read(wid__6, 'wid__6')
    print('malloc', sys.getsizeof(wid__6), 'wid__6')
    print(16, 16)
    arr__6 = instrument_read(np, 'np').random.rand(instrument_read(l__6,
        'l__6'), instrument_read(wid__6, 'wid__6'))
    write_instrument_read(arr__6, 'arr__6')
    print('malloc', sys.getsizeof(arr__6), 'arr__6')
    print(16, 17)
    neg__6 = True
    write_instrument_read(neg__6, 'neg__6')
    print('malloc', sys.getsizeof(neg__6), 'neg__6')
    for i__6 in range(instrument_read(l__6, 'l__6')):
        print('enter scope 7')
        for j__7 in range(instrument_read(wid__6, 'wid__6')):
            print('enter scope 8')
            print('enter scope 9')
            if instrument_read(neg__6, 'neg__6'):
                print(22, 21)
                arr__6[instrument_read(i__6, 'i__6')][instrument_read(j__7,
                    'j__7')] *= -1
                write_instrument_read_sub(arr__6[instrument_read(
                    instrument_read(i__6, 'i__6'), 'i__6')],
                    "arr__6[instrument_read(i__6, 'i__6')]",
                    instrument_read(instrument_read(j__7, 'j__7'), 'j__7'),
                    None, None, False)
            print('exit scope 9')
            print(23, 22)
            neg__6 = not instrument_read(neg__6, 'neg__6')
            write_instrument_read(neg__6, 'neg__6')
            print('malloc', sys.getsizeof(neg__6), 'neg__6')
            print('exit scope 8')
        print('exit scope 7')
    print('exit scope 6')
    return instrument_read(arr__6, 'arr__6')
    print('exit scope 6')


def arr_add(dst, src):
    print('enter scope 10')
    print(1, 25)
    dst__10 = instrument_read(dst, 'dst')
    write_instrument_read(dst__10, 'dst__10')
    print('malloc', sys.getsizeof(dst__10), 'dst__10')
    src__10 = instrument_read(src, 'src')
    write_instrument_read(src__10, 'src__10')
    print('malloc', sys.getsizeof(src__10), 'src__10')
    for i__10 in range(len(instrument_read(dst__10, 'dst__10'))):
        print('enter scope 11')
        for j__11 in range(len(instrument_read_sub(instrument_read(dst__10,
            'dst__10'), 'dst__10', 0, None, None, False))):
            print('enter scope 12')
            print(30, 28)
            dst__10[instrument_read(i__10, 'i__10')][instrument_read(j__11,
                'j__11')] += instrument_read_sub(instrument_read_sub(
                instrument_read(src__10, 'src__10'), 'src__10',
                instrument_read(i__10, 'i__10'), None, None, False),
                'src__10[i__10]', instrument_read(j__11, 'j__11'), None,
                None, False)
            write_instrument_read_sub(dst__10[instrument_read(
                instrument_read(i__10, 'i__10'), 'i__10')],
                "dst__10[instrument_read(i__10, 'i__10')]", instrument_read
                (instrument_read(j__11, 'j__11'), 'j__11'), None, None, False)
            print('exit scope 12')
        print('exit scope 11')
    print('exit scope 10')
    return instrument_read(dst__10, 'dst__10')
    print('exit scope 10')


def reLU(img):
    print('enter scope 13')
    print(1, 31)
    img__13 = instrument_read(img, 'img')
    write_instrument_read(img__13, 'img__13')
    print('malloc', sys.getsizeof(img__13), 'img__13')
    for i__13 in range(len(instrument_read(img__13, 'img__13'))):
        print('enter scope 14')
        for j__14 in range(len(instrument_read_sub(instrument_read(img__13,
            'img__13'), 'img__13', 0, None, None, False))):
            print('enter scope 15')
            print(38, 34)
            img__13[instrument_read(instrument_read(i__13, 'i__13'), 'i__13')][
                instrument_read(instrument_read(j__14, 'j__14'), 'j__14')
                ] = max(instrument_read_sub(instrument_read_sub(
                instrument_read(img__13, 'img__13'), 'img__13',
                instrument_read(i__13, 'i__13'), None, None, False),
                'img__13[i__13]', instrument_read(j__14, 'j__14'), None,
                None, False), 0)
            write_instrument_read_sub(img__13[instrument_read(
                instrument_read(i__13, 'i__13'), 'i__13')],
                "img__13[instrument_read(i__13, 'i__13')]", instrument_read
                (instrument_read(j__14, 'j__14'), 'j__14'), None, None, False)
            print('exit scope 15')
        print('exit scope 14')
    print('exit scope 13')
    return instrument_read(img__13, 'img__13')
    print('exit scope 13')


def get_mean(row):
    print('enter scope 16')
    print(1, 37)
    row__16 = instrument_read(row, 'row')
    write_instrument_read(row__16, 'row__16')
    print('malloc', sys.getsizeof(row__16), 'row__16')
    print(43, 38)
    sum_val__16 = 0
    write_instrument_read(sum_val__16, 'sum_val__16')
    print('malloc', sys.getsizeof(sum_val__16), 'sum_val__16')
    for i__16 in range(len(instrument_read(row__16, 'row__16'))):
        print('enter scope 17')
        print(45, 40)
        sum_val__16 += instrument_read_sub(instrument_read(row__16,
            'row__16'), 'row__16', instrument_read(i__16, 'i__16'), None,
            None, False)
        write_instrument_read(sum_val__16, 'sum_val__16')
        print('exit scope 17')
    print('exit scope 16')
    return instrument_read(sum_val__16, 'sum_val__16') / len(instrument_read
        (row__16, 'row__16'))
    print('exit scope 16')


def std_dev(row):
    print('enter scope 18')
    print(1, 43)
    row__18 = instrument_read(row, 'row')
    write_instrument_read(row__18, 'row__18')
    print('malloc', sys.getsizeof(row__18), 'row__18')
    print(50, 44)
    result__18 = 0
    write_instrument_read(result__18, 'result__18')
    print('malloc', sys.getsizeof(result__18), 'result__18')
    for i__18 in range(len(instrument_read(row__18, 'row__18'))):
        print('enter scope 19')
        print(52, 46)
        diff__19 = instrument_read_sub(instrument_read(row__18, 'row__18'),
            'row__18', instrument_read(i__18, 'i__18'), None, None, False
            ) - get_mean(instrument_read(row__18, 'row__18'))
        write_instrument_read(diff__19, 'diff__19')
        print('malloc', sys.getsizeof(diff__19), 'diff__19')
        print(52, 47)
        result__18 += instrument_read(diff__19, 'diff__19') * instrument_read(
            diff__19, 'diff__19')
        write_instrument_read(result__18, 'result__18')
        print('exit scope 19')
    print('exit scope 18')
    return instrument_read(math, 'math').sqrt(instrument_read(result__18,
        'result__18') / len(instrument_read(row__18, 'row__18')))
    print('exit scope 18')


def BN_layer(arr, weights, biases):
    print('enter scope 20')
    print(1, 50)
    arr__20 = instrument_read(arr, 'arr')
    write_instrument_read(arr__20, 'arr__20')
    print('malloc', sys.getsizeof(arr__20), 'arr__20')
    weights__20 = instrument_read(weights, 'weights')
    write_instrument_read(weights__20, 'weights__20')
    print('malloc', sys.getsizeof(weights__20), 'weights__20')
    biases__20 = instrument_read(biases, 'biases')
    write_instrument_read(biases__20, 'biases__20')
    print('malloc', sys.getsizeof(biases__20), 'biases__20')
    for i__20 in range(len(instrument_read(arr__20, 'arr__20'))):
        print('enter scope 21')
        print(58, 52)
        dev__21 = std_dev(instrument_read_sub(instrument_read(arr__20,
            'arr__20'), 'arr__20', instrument_read(i__20, 'i__20'), None,
            None, False))
        write_instrument_read(dev__21, 'dev__21')
        print('malloc', sys.getsizeof(dev__21), 'dev__21')
        print(58, 53)
        mean__21 = get_mean(instrument_read_sub(instrument_read(arr__20,
            'arr__20'), 'arr__20', instrument_read(i__20, 'i__20'), None,
            None, False))
        write_instrument_read(mean__21, 'mean__21')
        print('malloc', sys.getsizeof(mean__21), 'mean__21')
        print('enter scope 22')
        if instrument_read(dev__21, 'dev__21') == 0:
            print(60, 54)
            dev__21 = 1
            write_instrument_read(dev__21, 'dev__21')
            print('malloc', sys.getsizeof(dev__21), 'dev__21')
        print('exit scope 22')
        for j__21 in range(len(instrument_read_sub(instrument_read(arr__20,
            'arr__20'), 'arr__20', 0, None, None, False))):
            print('enter scope 23')
            print(62, 56)
            arr__20[instrument_read(instrument_read(i__20, 'i__20'), 'i__20')][
                instrument_read(instrument_read(j__21, 'j__21'), 'j__21')
                ] = instrument_read_sub(instrument_read(weights__20,
                'weights__20'), 'weights__20', instrument_read(i__20,
                'i__20'), None, None, False) * ((instrument_read_sub(
                instrument_read_sub(instrument_read(arr__20, 'arr__20'),
                'arr__20', instrument_read(i__20, 'i__20'), None, None,
                False), 'arr__20[i__20]', instrument_read(j__21, 'j__21'),
                None, None, False) - instrument_read(mean__21, 'mean__21')) /
                instrument_read(dev__21, 'dev__21')) + instrument_read_sub(
                instrument_read(biases__20, 'biases__20'), 'biases__20',
                instrument_read(i__20, 'i__20'), None, None, False)
            write_instrument_read_sub(arr__20[instrument_read(
                instrument_read(i__20, 'i__20'), 'i__20')],
                "arr__20[instrument_read(i__20, 'i__20')]", instrument_read
                (instrument_read(j__21, 'j__21'), 'j__21'), None, None, False)
            print('exit scope 23')
        print('exit scope 21')
    print('exit scope 20')
    return instrument_read(arr__20, 'arr__20')
    print('exit scope 20')


def fc_layer(arr, W, W_0):
    print('enter scope 24')
    print(1, 59)
    arr__24 = instrument_read(arr, 'arr')
    write_instrument_read(arr__24, 'arr__24')
    print('malloc', sys.getsizeof(arr__24), 'arr__24')
    W__24 = instrument_read(W, 'W')
    write_instrument_read(W__24, 'W__24')
    print('malloc', sys.getsizeof(W__24), 'W__24')
    W_0__24 = instrument_read(W_0, 'W_0')
    write_instrument_read(W_0__24, 'W_0__24')
    print('malloc', sys.getsizeof(W_0__24), 'W_0__24')
    print(67, 60)
    result__24 = instrument_read(np, 'np').zeros(len(instrument_read_sub(
        instrument_read(W__24, 'W__24'), 'W__24', 0, None, None, False)))
    write_instrument_read(result__24, 'result__24')
    print('malloc', sys.getsizeof(result__24), 'result__24')
    for i__24 in range(len(instrument_read_sub(instrument_read(W__24,
        'W__24'), 'W__24', 0, None, None, False))):
        print('enter scope 25')
        print(69, 62)
        sum_val__25 = instrument_read_sub(instrument_read(W_0__24,
            'W_0__24'), 'W_0__24', instrument_read(i__24, 'i__24'), None,
            None, False)
        write_instrument_read(sum_val__25, 'sum_val__25')
        print('malloc', sys.getsizeof(sum_val__25), 'sum_val__25')
        for j__25 in range(len(instrument_read(arr__24, 'arr__24'))):
            print('enter scope 26')
            print(72, 64)
            sum_val__25 += instrument_read_sub(instrument_read(arr__24,
                'arr__24'), 'arr__24', instrument_read(j__25, 'j__25'),
                None, None, False) * instrument_read_sub(instrument_read_sub
                (instrument_read(W__24, 'W__24'), 'W__24', instrument_read(
                j__25, 'j__25'), None, None, False), 'W__24[j__25]',
                instrument_read(i__24, 'i__24'), None, None, False)
            write_instrument_read(sum_val__25, 'sum_val__25')
            print('exit scope 26')
        print(73, 65)
        result__24[instrument_read(instrument_read(i__24, 'i__24'), 'i__24')
            ] = instrument_read(sum_val__25, 'sum_val__25')
        write_instrument_read_sub(result__24, 'result__24', instrument_read
            (instrument_read(i__24, 'i__24'), 'i__24'), None, None, False)
        print('exit scope 25')
    print('exit scope 24')
    return instrument_read(result__24, 'result__24')
    print('exit scope 24')


def softmax(arr):
    print('enter scope 27')
    print(1, 68)
    arr__27 = instrument_read(arr, 'arr')
    write_instrument_read(arr__27, 'arr__27')
    print('malloc', sys.getsizeof(arr__27), 'arr__27')
    print(77, 69)
    sum_val__27 = 0
    write_instrument_read(sum_val__27, 'sum_val__27')
    print('malloc', sys.getsizeof(sum_val__27), 'sum_val__27')
    for i__27 in range(len(instrument_read(arr__27, 'arr__27'))):
        print('enter scope 28')
        print(79, 70)
        sum_val__27 += instrument_read(math, 'math').exp(instrument_read_sub
            (instrument_read(arr__27, 'arr__27'), 'arr__27',
            instrument_read(i__27, 'i__27'), None, None, False))
        write_instrument_read(sum_val__27, 'sum_val__27')
        print('exit scope 28')
    print(80, 71)
    result__27 = instrument_read(np, 'np').zeros(len(instrument_read(
        arr__27, 'arr__27')))
    write_instrument_read(result__27, 'result__27')
    print('malloc', sys.getsizeof(result__27), 'result__27')
    for i__27 in range(len(instrument_read(arr__27, 'arr__27'))):
        print('enter scope 29')
        print(82, 72)
        result__27[instrument_read(instrument_read(i__27, 'i__27'), 'i__27')
            ] = instrument_read(math, 'math').exp(instrument_read_sub(
            instrument_read(arr__27, 'arr__27'), 'arr__27', instrument_read
            (i__27, 'i__27'), None, None, False)) / instrument_read(sum_val__27
            , 'sum_val__27')
        write_instrument_read_sub(result__27, 'result__27', instrument_read
            (instrument_read(i__27, 'i__27'), 'i__27'), None, None, False)
        print('exit scope 29')
    print('exit scope 27')
    return instrument_read(result__27, 'result__27')
    print('exit scope 27')


def concat(emb, head, tokens, d_k, cur):
    print('enter scope 30')
    print(1, 75)
    emb__30 = instrument_read(emb, 'emb')
    write_instrument_read(emb__30, 'emb__30')
    print('malloc', sys.getsizeof(emb__30), 'emb__30')
    head__30 = instrument_read(head, 'head')
    write_instrument_read(head__30, 'head__30')
    print('malloc', sys.getsizeof(head__30), 'head__30')
    tokens__30 = instrument_read(tokens, 'tokens')
    write_instrument_read(tokens__30, 'tokens__30')
    print('malloc', sys.getsizeof(tokens__30), 'tokens__30')
    d_k__30 = instrument_read(d_k, 'd_k')
    write_instrument_read(d_k__30, 'd_k__30')
    print('malloc', sys.getsizeof(d_k__30), 'd_k__30')
    cur__30 = instrument_read(cur, 'cur')
    write_instrument_read(cur__30, 'cur__30')
    print('malloc', sys.getsizeof(cur__30), 'cur__30')
    for i__30 in range(instrument_read(tokens__30, 'tokens__30')):
        print('enter scope 31')
        for j__31 in range(instrument_read(d_k__30, 'd_k__30')):
            print('enter scope 32')
            print(90, 78)
            emb__30[instrument_read(instrument_read(i__30, 'i__30'), 'i__30')][
                instrument_read(instrument_read(j__31, 'j__31'), 'j__31') +
                instrument_read(instrument_read(head__30, 'head__30'),
                'head__30') * instrument_read(instrument_read(d_k__30,
                'd_k__30'), 'd_k__30')] = instrument_read_sub(
                instrument_read_sub(instrument_read(cur__30, 'cur__30'),
                'cur__30', instrument_read(i__30, 'i__30'), None, None,
                False), 'cur__30[i__30]', instrument_read(j__31, 'j__31'),
                None, None, False)
            write_instrument_read_sub(emb__30[instrument_read(
                instrument_read(i__30, 'i__30'), 'i__30')],
                "emb__30[instrument_read(i__30, 'i__30')]", instrument_read
                (instrument_read(j__31, 'j__31'), 'j__31') + 
                instrument_read(instrument_read(head__30, 'head__30'),
                'head__30') * instrument_read(instrument_read(d_k__30,
                'd_k__30'), 'd_k__30'), None, None, False)
            print('exit scope 32')
        print('exit scope 31')
    print('exit scope 30')
    return instrument_read(emb__30, 'emb__30')
    print('exit scope 30')


def self_attn(head, tokens, d_k, Q, K, V):
    print('enter scope 33')
    print(1, 81)
    head__33 = instrument_read(head, 'head')
    write_instrument_read(head__33, 'head__33')
    print('malloc', sys.getsizeof(head__33), 'head__33')
    tokens__33 = instrument_read(tokens, 'tokens')
    write_instrument_read(tokens__33, 'tokens__33')
    print('malloc', sys.getsizeof(tokens__33), 'tokens__33')
    d_k__33 = instrument_read(d_k, 'd_k')
    write_instrument_read(d_k__33, 'd_k__33')
    print('malloc', sys.getsizeof(d_k__33), 'd_k__33')
    Q__33 = instrument_read(Q, 'Q')
    write_instrument_read(Q__33, 'Q__33')
    print('malloc', sys.getsizeof(Q__33), 'Q__33')
    K__33 = instrument_read(K, 'K')
    write_instrument_read(K__33, 'K__33')
    print('malloc', sys.getsizeof(K__33), 'K__33')
    V__33 = instrument_read(V, 'V')
    write_instrument_read(V__33, 'V__33')
    print('malloc', sys.getsizeof(V__33), 'V__33')
    print(95, 82)
    scores__33 = instrument_read(np, 'np').zeros((instrument_read(
        tokens__33, 'tokens__33'), instrument_read(tokens__33, 'tokens__33')))
    write_instrument_read(scores__33, 'scores__33')
    print('malloc', sys.getsizeof(scores__33), 'scores__33')
    for i__33 in range(instrument_read(tokens__33, 'tokens__33')):
        print('enter scope 34')
        for j__34 in range(instrument_read(tokens__33, 'tokens__33')):
            print('enter scope 35')
            for k__35 in range(instrument_read(d_k__33, 'd_k__33')):
                print('enter scope 36')
                print(101, 86)
                scores__33[instrument_read(i__33, 'i__33')][instrument_read
                    (j__34, 'j__34')] += instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (Q__33, 'Q__33'), 'Q__33', instrument_read(head__33,
                    'head__33'), None, None, False), 'Q__33[head__33]',
                    instrument_read(i__33, 'i__33'), None, None, False),
                    'Q__33[head__33][i__33]', instrument_read(k__35,
                    'k__35'), None, None, False) * instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (K__33, 'K__33'), 'K__33', instrument_read(head__33,
                    'head__33'), None, None, False), 'K__33[head__33]',
                    instrument_read(j__34, 'j__34'), None, None, False),
                    'K__33[head__33][j__34]', instrument_read(k__35,
                    'k__35'), None, None, False)
                write_instrument_read_sub(scores__33[instrument_read(
                    instrument_read(i__33, 'i__33'), 'i__33')],
                    "scores__33[instrument_read(i__33, 'i__33')]",
                    instrument_read(instrument_read(j__34, 'j__34'),
                    'j__34'), None, None, False)
                print('exit scope 36')
            print('exit scope 35')
        print('exit scope 34')
    for i__33 in range(instrument_read(tokens__33, 'tokens__33')):
        print('enter scope 37')
        for j__37 in range(instrument_read(tokens__33, 'tokens__33')):
            print('enter scope 38')
            print(105, 89)
            scores__33[instrument_read(i__33, 'i__33')][instrument_read(
                j__37, 'j__37')] /= instrument_read(math, 'math').sqrt(
                instrument_read(d_k__33, 'd_k__33'))
            write_instrument_read_sub(scores__33[instrument_read(
                instrument_read(i__33, 'i__33'), 'i__33')],
                "scores__33[instrument_read(i__33, 'i__33')]",
                instrument_read(instrument_read(j__37, 'j__37'), 'j__37'),
                None, None, False)
            print('exit scope 38')
        print(106, 91)
        scores__33 = instrument_read(np, 'np').random.rand(instrument_read(
            tokens__33, 'tokens__33'), instrument_read(tokens__33,
            'tokens__33'))
        write_instrument_read(scores__33, 'scores__33')
        print('malloc', sys.getsizeof(scores__33), 'scores__33')
        print(106, 92)
        scores__33[instrument_read(instrument_read(i__33, 'i__33'), 'i__33')
            ] = softmax(instrument_read_sub(instrument_read(scores__33,
            'scores__33'), 'scores__33', instrument_read(i__33, 'i__33'),
            None, None, False))
        write_instrument_read_sub(scores__33, 'scores__33', instrument_read
            (instrument_read(i__33, 'i__33'), 'i__33'), None, None, False)
        print('exit scope 37')
    print(104, 93)
    out__33 = instrument_read(np, 'np').zeros((instrument_read(tokens__33,
        'tokens__33'), instrument_read(d_k__33, 'd_k__33')))
    write_instrument_read(out__33, 'out__33')
    print('malloc', sys.getsizeof(out__33), 'out__33')
    for i__33 in range(instrument_read(tokens__33, 'tokens__33')):
        print('enter scope 39')
        for j__39 in range(instrument_read(d_k__33, 'd_k__33')):
            print('enter scope 40')
            for k__40 in range(instrument_read(tokens__33, 'tokens__33')):
                print('enter scope 41')
                print(112, 97)
                out__33[instrument_read(i__33, 'i__33')][instrument_read(
                    j__39, 'j__39')] += instrument_read_sub(instrument_read_sub
                    (instrument_read(scores__33, 'scores__33'),
                    'scores__33', instrument_read(i__33, 'i__33'), None,
                    None, False), 'scores__33[i__33]', instrument_read(
                    k__40, 'k__40'), None, None, False) * instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (V__33, 'V__33'), 'V__33', instrument_read(head__33,
                    'head__33'), None, None, False), 'V__33[head__33]',
                    instrument_read(k__40, 'k__40'), None, None, False),
                    'V__33[head__33][k__40]', instrument_read(j__39,
                    'j__39'), None, None, False)
                write_instrument_read_sub(out__33[instrument_read(
                    instrument_read(i__33, 'i__33'), 'i__33')],
                    "out__33[instrument_read(i__33, 'i__33')]",
                    instrument_read(instrument_read(j__39, 'j__39'),
                    'j__39'), None, None, False)
                print('exit scope 41')
            print('exit scope 40')
        print('exit scope 39')
    print('exit scope 33')
    return instrument_read(out__33, 'out__33')
    print('exit scope 33')


def main():
    print('enter scope 42')
    print(1, 101)
    print(117, 102)
    d_model__42, heads__42, tokens__42, layers__42 = 12, 12, 8, 12
    write_instrument_read(layers__42, 'layers__42')
    print('malloc', sys.getsizeof(layers__42), 'layers__42')
    print(117, 103)
    d_k__42 = instrument_read(d_model__42, 'd_model__42') // instrument_read(
        heads__42, 'heads__42')
    write_instrument_read(d_k__42, 'd_k__42')
    print('malloc', sys.getsizeof(d_k__42), 'd_k__42')
    print(117, 104)
    embeddings__42 = instrument_read(np, 'np').random.rand(instrument_read(
        tokens__42, 'tokens__42'), instrument_read(d_model__42, 'd_model__42'))
    write_instrument_read(embeddings__42, 'embeddings__42')
    print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
    for i__42 in range(instrument_read(tokens__42, 'tokens__42')):
        print('enter scope 43')
        for j__43 in range(instrument_read(d_model__42, 'd_model__42')):
            print('enter scope 44')
            print('enter scope 45')
            if instrument_read(j__43, 'j__43') % 2 == 0:
                print(123, 108)
                embeddings__42[instrument_read(i__42, 'i__42')][instrument_read
                    (j__43, 'j__43')] += instrument_read(math, 'math').sin(
                    instrument_read(i__42, 'i__42') / instrument_read(math,
                    'math').pow(10000, 2 * instrument_read(j__43, 'j__43') /
                    instrument_read(d_model__42, 'd_model__42')))
                write_instrument_read_sub(embeddings__42[instrument_read(
                    instrument_read(i__42, 'i__42'), 'i__42')],
                    "embeddings__42[instrument_read(i__42, 'i__42')]",
                    instrument_read(instrument_read(j__43, 'j__43'),
                    'j__43'), None, None, False)
            else:
                print(125, 110)
                embeddings__42[instrument_read(i__42, 'i__42')][instrument_read
                    (j__43, 'j__43')] += instrument_read(math, 'math').cos(
                    instrument_read(i__42, 'i__42') / instrument_read(math,
                    'math').pow(10000, 2 * instrument_read(j__43, 'j__43') /
                    instrument_read(d_model__42, 'd_model__42')))
                write_instrument_read_sub(embeddings__42[instrument_read(
                    instrument_read(i__42, 'i__42'), 'i__42')],
                    "embeddings__42[instrument_read(i__42, 'i__42')]",
                    instrument_read(instrument_read(j__43, 'j__43'),
                    'j__43'), None, None, False)
            print('exit scope 45')
            print('exit scope 44')
        print('exit scope 43')
    print(120, 111)
    W_Q__42 = balance_random_3d(instrument_read(heads__42, 'heads__42'),
        instrument_read(d_model__42, 'd_model__42'), instrument_read(
        d_k__42, 'd_k__42'))
    write_instrument_read(W_Q__42, 'W_Q__42')
    print('malloc', sys.getsizeof(W_Q__42), 'W_Q__42')
    print(120, 112)
    W_K__42 = balance_random_3d(instrument_read(heads__42, 'heads__42'),
        instrument_read(d_model__42, 'd_model__42'), instrument_read(
        d_k__42, 'd_k__42'))
    write_instrument_read(W_K__42, 'W_K__42')
    print('malloc', sys.getsizeof(W_K__42), 'W_K__42')
    print(120, 113)
    W_V__42 = balance_random_3d(instrument_read(heads__42, 'heads__42'),
        instrument_read(d_model__42, 'd_model__42'), instrument_read(
        d_k__42, 'd_k__42'))
    write_instrument_read(W_V__42, 'W_V__42')
    print('malloc', sys.getsizeof(W_V__42), 'W_V__42')
    print(120, 114)
    Q__42 = instrument_read(np, 'np').zeros((instrument_read(heads__42,
        'heads__42'), instrument_read(tokens__42, 'tokens__42'),
        instrument_read(d_k__42, 'd_k__42')))
    write_instrument_read(Q__42, 'Q__42')
    print('malloc', sys.getsizeof(Q__42), 'Q__42')
    print(120, 115)
    K__42 = instrument_read(np, 'np').zeros((instrument_read(heads__42,
        'heads__42'), instrument_read(tokens__42, 'tokens__42'),
        instrument_read(d_k__42, 'd_k__42')))
    write_instrument_read(K__42, 'K__42')
    print('malloc', sys.getsizeof(K__42), 'K__42')
    print(120, 116)
    V__42 = instrument_read(np, 'np').zeros((instrument_read(heads__42,
        'heads__42'), instrument_read(tokens__42, 'tokens__42'),
        instrument_read(d_k__42, 'd_k__42')))
    write_instrument_read(V__42, 'V__42')
    print('malloc', sys.getsizeof(V__42), 'V__42')
    for i__42 in range(instrument_read(heads__42, 'heads__42')):
        print('enter scope 46')
        for j__46 in range(instrument_read(tokens__42, 'tokens__42')):
            print('enter scope 47')
            for k__47 in range(instrument_read(d_k__42, 'd_k__42')):
                print('enter scope 48')
                print(131, 120)
                sumQ__48, sumK__48, sumV__48 = 0, 0, 0
                write_instrument_read(sumV__48, 'sumV__48')
                print('malloc', sys.getsizeof(sumV__48), 'sumV__48')
                for a__48 in range(instrument_read(d_model__42, 'd_model__42')
                    ):
                    print('enter scope 49')
                    print(134, 122)
                    sumQ__48 += instrument_read_sub(instrument_read_sub(
                        instrument_read(embeddings__42, 'embeddings__42'),
                        'embeddings__42', instrument_read(j__46, 'j__46'),
                        None, None, False), 'embeddings__42[j__46]',
                        instrument_read(a__48, 'a__48'), None, None, False
                        ) * instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(W_Q__42,
                        'W_Q__42'), 'W_Q__42', instrument_read(i__42,
                        'i__42'), None, None, False), 'W_Q__42[i__42]',
                        instrument_read(a__48, 'a__48'), None, None, False),
                        'W_Q__42[i__42][a__48]', instrument_read(k__47,
                        'k__47'), None, None, False)
                    write_instrument_read(sumQ__48, 'sumQ__48')
                    print(134, 123)
                    sumK__48 += instrument_read_sub(instrument_read_sub(
                        instrument_read(embeddings__42, 'embeddings__42'),
                        'embeddings__42', instrument_read(j__46, 'j__46'),
                        None, None, False), 'embeddings__42[j__46]',
                        instrument_read(a__48, 'a__48'), None, None, False
                        ) * instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(W_K__42,
                        'W_K__42'), 'W_K__42', instrument_read(i__42,
                        'i__42'), None, None, False), 'W_K__42[i__42]',
                        instrument_read(a__48, 'a__48'), None, None, False),
                        'W_K__42[i__42][a__48]', instrument_read(k__47,
                        'k__47'), None, None, False)
                    write_instrument_read(sumK__48, 'sumK__48')
                    print(134, 124)
                    sumV__48 += instrument_read_sub(instrument_read_sub(
                        instrument_read(embeddings__42, 'embeddings__42'),
                        'embeddings__42', instrument_read(j__46, 'j__46'),
                        None, None, False), 'embeddings__42[j__46]',
                        instrument_read(a__48, 'a__48'), None, None, False
                        ) * instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(W_V__42,
                        'W_V__42'), 'W_V__42', instrument_read(i__42,
                        'i__42'), None, None, False), 'W_V__42[i__42]',
                        instrument_read(a__48, 'a__48'), None, None, False),
                        'W_V__42[i__42][a__48]', instrument_read(k__47,
                        'k__47'), None, None, False)
                    write_instrument_read(sumV__48, 'sumV__48')
                    print('exit scope 49')
                print(135, 125)
                Q__42[instrument_read(instrument_read(i__42, 'i__42'), 'i__42')
                    ][instrument_read(instrument_read(j__46, 'j__46'), 'j__46')
                    ][instrument_read(instrument_read(k__47, 'k__47'), 'k__47')
                    ] = instrument_read(sumQ__48, 'sumQ__48')
                write_instrument_read_sub(Q__42[instrument_read(
                    instrument_read(i__42, 'i__42'), 'i__42')][
                    instrument_read(instrument_read(j__46, 'j__46'),
                    'j__46')],
                    "Q__42[instrument_read(i__42, 'i__42')][instrument_read(j__46, 'j__46')]"
                    , instrument_read(instrument_read(k__47, 'k__47'),
                    'k__47'), None, None, False)
                print(135, 126)
                K__42[instrument_read(instrument_read(i__42, 'i__42'), 'i__42')
                    ][instrument_read(instrument_read(j__46, 'j__46'), 'j__46')
                    ][instrument_read(instrument_read(k__47, 'k__47'), 'k__47')
                    ] = instrument_read(sumK__48, 'sumK__48')
                write_instrument_read_sub(K__42[instrument_read(
                    instrument_read(i__42, 'i__42'), 'i__42')][
                    instrument_read(instrument_read(j__46, 'j__46'),
                    'j__46')],
                    "K__42[instrument_read(i__42, 'i__42')][instrument_read(j__46, 'j__46')]"
                    , instrument_read(instrument_read(k__47, 'k__47'),
                    'k__47'), None, None, False)
                print(135, 127)
                V__42[instrument_read(instrument_read(i__42, 'i__42'), 'i__42')
                    ][instrument_read(instrument_read(j__46, 'j__46'), 'j__46')
                    ][instrument_read(instrument_read(k__47, 'k__47'), 'k__47')
                    ] = instrument_read(sumV__48, 'sumV__48')
                write_instrument_read_sub(V__42[instrument_read(
                    instrument_read(i__42, 'i__42'), 'i__42')][
                    instrument_read(instrument_read(j__46, 'j__46'),
                    'j__46')],
                    "V__42[instrument_read(i__42, 'i__42')][instrument_read(j__46, 'j__46')]"
                    , instrument_read(instrument_read(k__47, 'k__47'),
                    'k__47'), None, None, False)
                print('exit scope 48')
            print('exit scope 47')
        print('exit scope 46')
    for i__42 in range(instrument_read(layers__42, 'layers__42')):
        print('enter scope 50')
        print(136, 129)
        emb_cpy__50 = instrument_read(np, 'np').copy(instrument_read(
            embeddings__42, 'embeddings__42'))
        write_instrument_read(emb_cpy__50, 'emb_cpy__50')
        print('malloc', sys.getsizeof(emb_cpy__50), 'emb_cpy__50')
        print(136, 130)
        multi_head_out__50 = instrument_read(np, 'np').zeros((
            instrument_read(tokens__42, 'tokens__42'), instrument_read(
            d_model__42, 'd_model__42')))
        write_instrument_read(multi_head_out__50, 'multi_head_out__50')
        print('malloc', sys.getsizeof(multi_head_out__50), 'multi_head_out__50'
            )
        for j__50 in range(instrument_read(heads__42, 'heads__42')):
            print('enter scope 51')
            print(139, 132)
            cur__51 = self_attn(instrument_read(j__50, 'j__50'),
                instrument_read(tokens__42, 'tokens__42'), instrument_read(
                d_k__42, 'd_k__42'), instrument_read(Q__42, 'Q__42'),
                instrument_read(K__42, 'K__42'), instrument_read(V__42,
                'V__42'))
            write_instrument_read(cur__51, 'cur__51')
            print('malloc', sys.getsizeof(cur__51), 'cur__51')
            print(139, 133)
            multi_head_out__50 = concat(instrument_read(multi_head_out__50,
                'multi_head_out__50'), instrument_read(j__50, 'j__50'),
                instrument_read(tokens__42, 'tokens__42'), instrument_read(
                d_k__42, 'd_k__42'), instrument_read(cur__51, 'cur__51'))
            write_instrument_read(multi_head_out__50, 'multi_head_out__50')
            print('malloc', sys.getsizeof(multi_head_out__50),
                'multi_head_out__50')
            print('exit scope 51')
        print(140, 134)
        W_attn__50 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model__42, 'd_model__42'), instrument_read(d_model__42,
            'd_model__42'))
        write_instrument_read(W_attn__50, 'W_attn__50')
        print('malloc', sys.getsizeof(W_attn__50), 'W_attn__50')
        for i__42 in range(instrument_read(tokens__42, 'tokens__42')):
            print('enter scope 52')
            for j__50 in range(instrument_read(d_model__42, 'd_model__42')):
                print('enter scope 53')
                print(144, 137)
                sum_val__53 = 0
                write_instrument_read(sum_val__53, 'sum_val__53')
                print('malloc', sys.getsizeof(sum_val__53), 'sum_val__53')
                for k__53 in range(instrument_read(d_model__42, 'd_model__42')
                    ):
                    print('enter scope 54')
                    print(147, 139)
                    sum_val__53 += instrument_read_sub(instrument_read_sub(
                        instrument_read(multi_head_out__50,
                        'multi_head_out__50'), 'multi_head_out__50',
                        instrument_read(i__42, 'i__42'), None, None, False),
                        'multi_head_out__50[i__42]', instrument_read(k__53,
                        'k__53'), None, None, False) * instrument_read_sub(
                        instrument_read_sub(instrument_read(W_attn__50,
                        'W_attn__50'), 'W_attn__50', instrument_read(k__53,
                        'k__53'), None, None, False), 'W_attn__50[k__53]',
                        instrument_read(j__50, 'j__50'), None, None, False)
                    write_instrument_read(sum_val__53, 'sum_val__53')
                    print('exit scope 54')
                print(148, 140)
                embeddings__42[instrument_read(instrument_read(i__42,
                    'i__42'), 'i__42')][instrument_read(instrument_read(
                    j__50, 'j__50'), 'j__50')] = instrument_read(sum_val__53,
                    'sum_val__53')
                write_instrument_read_sub(embeddings__42[instrument_read(
                    instrument_read(i__42, 'i__42'), 'i__42')],
                    "embeddings__42[instrument_read(i__42, 'i__42')]",
                    instrument_read(instrument_read(j__50, 'j__50'),
                    'j__50'), None, None, False)
                print('exit scope 53')
            print('exit scope 52')
        print(143, 141)
        embeddings__42 = arr_add(instrument_read(embeddings__42,
            'embeddings__42'), instrument_read(emb_cpy__50, 'emb_cpy__50'))
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print(143, 142)
        weights__50, biases__50 = instrument_read(np, 'np').random.rand(
            instrument_read(d_model__42, 'd_model__42')), instrument_read(np,
            'np').random.rand(instrument_read(d_model__42, 'd_model__42'))
        write_instrument_read(biases__50, 'biases__50')
        print('malloc', sys.getsizeof(biases__50), 'biases__50')
        print(143, 143)
        embeddings__42 = BN_layer(instrument_read(embeddings__42,
            'embeddings__42'), instrument_read(weights__50, 'weights__50'),
            instrument_read(biases__50, 'biases__50'))
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print(143, 144)
        emb_cpy__50 = instrument_read(np, 'np').copy(instrument_read(
            embeddings__42, 'embeddings__42'))
        write_instrument_read(emb_cpy__50, 'emb_cpy__50')
        print('malloc', sys.getsizeof(emb_cpy__50), 'emb_cpy__50')
        print(143, 145)
        W__50 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model__42, 'd_model__42'), instrument_read(d_model__42,
            'd_model__42') * 4)
        write_instrument_read(W__50, 'W__50')
        print('malloc', sys.getsizeof(W__50), 'W__50')
        print(143, 146)
        W_0__50 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model__42, 'd_model__42') * 4)
        write_instrument_read(W_0__50, 'W_0__50')
        print('malloc', sys.getsizeof(W_0__50), 'W_0__50')
        print(143, 147)
        emb_new__50 = instrument_read(np, 'np').zeros((instrument_read(
            tokens__42, 'tokens__42'), instrument_read(d_model__42,
            'd_model__42') * 4))
        write_instrument_read(emb_new__50, 'emb_new__50')
        print('malloc', sys.getsizeof(emb_new__50), 'emb_new__50')
        for i__42 in range(instrument_read(tokens__42, 'tokens__42')):
            print('enter scope 55')
            print(150, 149)
            emb_new__50[instrument_read(instrument_read(i__42, 'i__42'),
                'i__42')] = fc_layer(instrument_read_sub(instrument_read(
                embeddings__42, 'embeddings__42'), 'embeddings__42',
                instrument_read(i__42, 'i__42'), None, None, False),
                instrument_read(W__50, 'W__50'), instrument_read(W_0__50,
                'W_0__50'))
            write_instrument_read_sub(emb_new__50, 'emb_new__50',
                instrument_read(instrument_read(i__42, 'i__42'), 'i__42'),
                None, None, False)
            print('exit scope 55')
        print(151, 150)
        embeddings__42 = instrument_read(emb_new__50, 'emb_new__50')
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print(151, 151)
        embeddings__42 = reLU(instrument_read(embeddings__42, 'embeddings__42')
            )
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print(151, 152)
        W__50 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model__42, 'd_model__42') * 4, instrument_read(d_model__42,
            'd_model__42'))
        write_instrument_read(W__50, 'W__50')
        print('malloc', sys.getsizeof(W__50), 'W__50')
        print(151, 153)
        W_0__50 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model__42, 'd_model__42'))
        write_instrument_read(W_0__50, 'W_0__50')
        print('malloc', sys.getsizeof(W_0__50), 'W_0__50')
        print(151, 154)
        emb_new__50 = instrument_read(np, 'np').zeros((instrument_read(
            tokens__42, 'tokens__42'), instrument_read(d_model__42,
            'd_model__42')))
        write_instrument_read(emb_new__50, 'emb_new__50')
        print('malloc', sys.getsizeof(emb_new__50), 'emb_new__50')
        for i__42 in range(instrument_read(tokens__42, 'tokens__42')):
            print('enter scope 56')
            print(153, 156)
            emb_new__50[instrument_read(instrument_read(i__42, 'i__42'),
                'i__42')] = fc_layer(instrument_read_sub(instrument_read(
                embeddings__42, 'embeddings__42'), 'embeddings__42',
                instrument_read(i__42, 'i__42'), None, None, False),
                instrument_read(W__50, 'W__50'), instrument_read(W_0__50,
                'W_0__50'))
            write_instrument_read_sub(emb_new__50, 'emb_new__50',
                instrument_read(instrument_read(i__42, 'i__42'), 'i__42'),
                None, None, False)
            print('exit scope 56')
        print(154, 157)
        embeddings__42 = instrument_read(emb_new__50, 'emb_new__50')
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print(154, 158)
        embeddings__42 = arr_add(instrument_read(embeddings__42,
            'embeddings__42'), instrument_read(emb_cpy__50, 'emb_cpy__50'))
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print(154, 159)
        embeddings__42 = BN_layer(instrument_read(embeddings__42,
            'embeddings__42'), instrument_read(weights__50, 'weights__50'),
            instrument_read(biases__50, 'biases__50'))
        write_instrument_read(embeddings__42, 'embeddings__42')
        print('malloc', sys.getsizeof(embeddings__42), 'embeddings__42')
        print('exit scope 50')
    print('exit scope 42')


print('enter scope 57')
if instrument_read(__name__, '__name__') == '__main__':
    main()
print('exit scope 57')
