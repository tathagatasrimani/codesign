import sys
from instrument_lib import *
import sys
from instrument_lib import *
import numpy as np
import math
from loop import loop


def balance_random_3d(depth, l, wid):
    print('enter scope 1')
    print(1, 5)
    depth_1 = instrument_read(depth, 'depth')
    write_instrument_read(depth_1, 'depth_1')
    print('malloc', sys.getsizeof(depth_1), 'depth_1')
    l_1 = instrument_read(l, 'l')
    write_instrument_read(l_1, 'l_1')
    print('malloc', sys.getsizeof(l_1), 'l_1')
    wid_1 = instrument_read(wid, 'wid')
    write_instrument_read(wid_1, 'wid_1')
    print('malloc', sys.getsizeof(wid_1), 'wid_1')
    print(3, 6)
    arr_1 = instrument_read(np, 'np').random.rand(instrument_read(depth_1,
        'depth_1'), instrument_read(l_1, 'l_1'), instrument_read(wid_1,
        'wid_1'))
    write_instrument_read(arr_1, 'arr_1')
    print('malloc', sys.getsizeof(arr_1), 'arr_1')
    print(3, 7)
    neg_1 = True
    write_instrument_read(neg_1, 'neg_1')
    print('malloc', sys.getsizeof(neg_1), 'neg_1')
    for i_1 in range(instrument_read(depth_1, 'depth_1')):
        for j_1 in range(instrument_read(l_1, 'l_1')):
            for k_1 in range(instrument_read(wid_1, 'wid_1')):
                if instrument_read(neg_1, 'neg_1'):
                    print(11, 12)
                    arr_1[instrument_read(i_1, 'i_1')][instrument_read(j_1,
                        'j_1')][instrument_read(k_1, 'k_1')] *= -1
                    write_instrument_read_sub(arr_1[instrument_read(
                        instrument_read(i_1, 'i_1'), 'i_1')][
                        instrument_read(instrument_read(j_1, 'j_1'), 'j_1')
                        ],
                        "arr_1[instrument_read(i_1, 'i_1')][instrument_read(j_1, 'j_1')]"
                        , instrument_read(instrument_read(k_1, 'k_1'),
                        'k_1'), None, None, False)
                print(12, 13)
                neg_1 = not instrument_read(neg_1, 'neg_1')
                write_instrument_read(neg_1, 'neg_1')
                print('malloc', sys.getsizeof(neg_1), 'neg_1')
    print('exit scope 1')
    return instrument_read(arr_1, 'arr_1')
    print('exit scope 1')


def balance_random_2d(l, wid):
    print('enter scope 2')
    print(1, 16)
    l_2 = instrument_read(l, 'l')
    write_instrument_read(l_2, 'l_2')
    print('malloc', sys.getsizeof(l_2), 'l_2')
    wid_2 = instrument_read(wid, 'wid')
    write_instrument_read(wid_2, 'wid_2')
    print('malloc', sys.getsizeof(wid_2), 'wid_2')
    print(16, 17)
    arr_2 = instrument_read(np, 'np').random.rand(instrument_read(l_2,
        'l_2'), instrument_read(wid_2, 'wid_2'))
    write_instrument_read(arr_2, 'arr_2')
    print('malloc', sys.getsizeof(arr_2), 'arr_2')
    print(16, 18)
    neg_2 = True
    write_instrument_read(neg_2, 'neg_2')
    print('malloc', sys.getsizeof(neg_2), 'neg_2')
    for i_2 in range(instrument_read(l_2, 'l_2')):
        for j_2 in range(instrument_read(wid_2, 'wid_2')):
            if instrument_read(neg_2, 'neg_2'):
                print(22, 22)
                arr_2[instrument_read(i_2, 'i_2')][instrument_read(j_2, 'j_2')
                    ] *= -1
                write_instrument_read_sub(arr_2[instrument_read(
                    instrument_read(i_2, 'i_2'), 'i_2')],
                    "arr_2[instrument_read(i_2, 'i_2')]", instrument_read(
                    instrument_read(j_2, 'j_2'), 'j_2'), None, None, False)
            print(23, 23)
            neg_2 = not instrument_read(neg_2, 'neg_2')
            write_instrument_read(neg_2, 'neg_2')
            print('malloc', sys.getsizeof(neg_2), 'neg_2')
    print('exit scope 2')
    return instrument_read(arr_2, 'arr_2')
    print('exit scope 2')


def arr_add(dst, src):
    print('enter scope 3')
    print(1, 26)
    dst_3 = instrument_read(dst, 'dst')
    write_instrument_read(dst_3, 'dst_3')
    print('malloc', sys.getsizeof(dst_3), 'dst_3')
    src_3 = instrument_read(src, 'src')
    write_instrument_read(src_3, 'src_3')
    print('malloc', sys.getsizeof(src_3), 'src_3')
    for i_3 in range(len(instrument_read(dst_3, 'dst_3'))):
        for j_3 in range(len(instrument_read_sub(instrument_read(dst_3,
            'dst_3'), 'dst_3', 0, None, None, False))):
            print(30, 29)
            dst_3[instrument_read(i_3, 'i_3')][instrument_read(j_3, 'j_3')
                ] += instrument_read_sub(instrument_read_sub(
                instrument_read(src_3, 'src_3'), 'src_3', instrument_read(
                i_3, 'i_3'), None, None, False), 'src_3[i_3]',
                instrument_read(j_3, 'j_3'), None, None, False)
            write_instrument_read_sub(dst_3[instrument_read(instrument_read
                (i_3, 'i_3'), 'i_3')], "dst_3[instrument_read(i_3, 'i_3')]",
                instrument_read(instrument_read(j_3, 'j_3'), 'j_3'), None,
                None, False)
    print('exit scope 3')
    return instrument_read(dst_3, 'dst_3')
    print('exit scope 3')


def reLU(img):
    print('enter scope 4')
    print(1, 32)
    img_4 = instrument_read(img, 'img')
    write_instrument_read(img_4, 'img_4')
    print('malloc', sys.getsizeof(img_4), 'img_4')
    for i_4 in range(len(instrument_read(img_4, 'img_4'))):
        for j_4 in range(len(instrument_read_sub(instrument_read(img_4,
            'img_4'), 'img_4', 0, None, None, False))):
            print(38, 35)
            img_4[instrument_read(instrument_read(i_4, 'i_4'), 'i_4')][
                instrument_read(instrument_read(j_4, 'j_4'), 'j_4')] = max(
                instrument_read_sub(instrument_read_sub(instrument_read(
                img_4, 'img_4'), 'img_4', instrument_read(i_4, 'i_4'), None,
                None, False), 'img_4[i_4]', instrument_read(j_4, 'j_4'),
                None, None, False), 0)
            write_instrument_read_sub(img_4[instrument_read(instrument_read
                (i_4, 'i_4'), 'i_4')], "img_4[instrument_read(i_4, 'i_4')]",
                instrument_read(instrument_read(j_4, 'j_4'), 'j_4'), None,
                None, False)
    print('exit scope 4')
    return instrument_read(img_4, 'img_4')
    print('exit scope 4')


def get_mean(row):
    print('enter scope 5')
    print(1, 38)
    row_5 = instrument_read(row, 'row')
    write_instrument_read(row_5, 'row_5')
    print('malloc', sys.getsizeof(row_5), 'row_5')
    print(43, 39)
    sum_val_5 = 0
    write_instrument_read(sum_val_5, 'sum_val_5')
    print('malloc', sys.getsizeof(sum_val_5), 'sum_val_5')
    for i_5 in range(len(instrument_read(row_5, 'row_5'))):
        print(45, 41)
        sum_val_5 += instrument_read_sub(instrument_read(row_5, 'row_5'),
            'row_5', instrument_read(i_5, 'i_5'), None, None, False)
        write_instrument_read(sum_val_5, 'sum_val_5')
    print('exit scope 5')
    return instrument_read(sum_val_5, 'sum_val_5') / len(instrument_read(
        row_5, 'row_5'))
    print('exit scope 5')


def std_dev(row):
    print('enter scope 6')
    print(1, 44)
    row_6 = instrument_read(row, 'row')
    write_instrument_read(row_6, 'row_6')
    print('malloc', sys.getsizeof(row_6), 'row_6')
    print(50, 45)
    result_6 = 0
    write_instrument_read(result_6, 'result_6')
    print('malloc', sys.getsizeof(result_6), 'result_6')
    for i_6 in range(len(instrument_read(row_6, 'row_6'))):
        print(52, 47)
        diff_6 = instrument_read_sub(instrument_read(row_6, 'row_6'),
            'row_6', instrument_read(i_6, 'i_6'), None, None, False
            ) - get_mean(instrument_read(row_6, 'row_6'))
        write_instrument_read(diff_6, 'diff_6')
        print('malloc', sys.getsizeof(diff_6), 'diff_6')
        print(52, 48)
        result_6 += instrument_read(diff_6, 'diff_6') * instrument_read(diff_6,
            'diff_6')
        write_instrument_read(result_6, 'result_6')
    print('exit scope 6')
    return instrument_read(math, 'math').sqrt(instrument_read(result_6,
        'result_6') / len(instrument_read(row_6, 'row_6')))
    print('exit scope 6')


def BN_layer(arr, weights, biases):
    print('enter scope 7')
    print(1, 51)
    arr_7 = instrument_read(arr, 'arr')
    write_instrument_read(arr_7, 'arr_7')
    print('malloc', sys.getsizeof(arr_7), 'arr_7')
    weights_7 = instrument_read(weights, 'weights')
    write_instrument_read(weights_7, 'weights_7')
    print('malloc', sys.getsizeof(weights_7), 'weights_7')
    biases_7 = instrument_read(biases, 'biases')
    write_instrument_read(biases_7, 'biases_7')
    print('malloc', sys.getsizeof(biases_7), 'biases_7')
    for i_7 in range(len(instrument_read(arr_7, 'arr_7'))):
        print(58, 53)
        dev_7 = std_dev(instrument_read_sub(instrument_read(arr_7, 'arr_7'),
            'arr_7', instrument_read(i_7, 'i_7'), None, None, False))
        write_instrument_read(dev_7, 'dev_7')
        print('malloc', sys.getsizeof(dev_7), 'dev_7')
        print(58, 54)
        mean_7 = get_mean(instrument_read_sub(instrument_read(arr_7,
            'arr_7'), 'arr_7', instrument_read(i_7, 'i_7'), None, None, False))
        write_instrument_read(mean_7, 'mean_7')
        print('malloc', sys.getsizeof(mean_7), 'mean_7')
        if instrument_read(dev_7, 'dev_7') == 0:
            print(60, 55)
            dev_7 = 1
            write_instrument_read(dev_7, 'dev_7')
            print('malloc', sys.getsizeof(dev_7), 'dev_7')
        for j_7 in range(len(instrument_read_sub(instrument_read(arr_7,
            'arr_7'), 'arr_7', 0, None, None, False))):
            print(62, 57)
            arr_7[instrument_read(instrument_read(i_7, 'i_7'), 'i_7')][
                instrument_read(instrument_read(j_7, 'j_7'), 'j_7')
                ] = instrument_read_sub(instrument_read(weights_7,
                'weights_7'), 'weights_7', instrument_read(i_7, 'i_7'),
                None, None, False) * ((instrument_read_sub(
                instrument_read_sub(instrument_read(arr_7, 'arr_7'),
                'arr_7', instrument_read(i_7, 'i_7'), None, None, False),
                'arr_7[i_7]', instrument_read(j_7, 'j_7'), None, None,
                False) - instrument_read(mean_7, 'mean_7')) /
                instrument_read(dev_7, 'dev_7')) + instrument_read_sub(
                instrument_read(biases_7, 'biases_7'), 'biases_7',
                instrument_read(i_7, 'i_7'), None, None, False)
            write_instrument_read_sub(arr_7[instrument_read(instrument_read
                (i_7, 'i_7'), 'i_7')], "arr_7[instrument_read(i_7, 'i_7')]",
                instrument_read(instrument_read(j_7, 'j_7'), 'j_7'), None,
                None, False)
    print('exit scope 7')
    return instrument_read(arr_7, 'arr_7')
    print('exit scope 7')


def fc_layer(arr, W, W_0):
    print('enter scope 8')
    print(1, 60)
    arr_8 = instrument_read(arr, 'arr')
    write_instrument_read(arr_8, 'arr_8')
    print('malloc', sys.getsizeof(arr_8), 'arr_8')
    W_8 = instrument_read(W, 'W')
    write_instrument_read(W_8, 'W_8')
    print('malloc', sys.getsizeof(W_8), 'W_8')
    W_0_8 = instrument_read(W_0, 'W_0')
    write_instrument_read(W_0_8, 'W_0_8')
    print('malloc', sys.getsizeof(W_0_8), 'W_0_8')
    print(67, 61)
    result_8 = instrument_read(np, 'np').zeros(len(instrument_read_sub(
        instrument_read(W_8, 'W_8'), 'W_8', 0, None, None, False)))
    write_instrument_read(result_8, 'result_8')
    print('malloc', sys.getsizeof(result_8), 'result_8')
    for i_8 in range(len(instrument_read_sub(instrument_read(W_8, 'W_8'),
        'W_8', 0, None, None, False))):
        print(69, 63)
        sum_val_8 = instrument_read_sub(instrument_read(W_0_8, 'W_0_8'),
            'W_0_8', instrument_read(i_8, 'i_8'), None, None, False)
        write_instrument_read(sum_val_8, 'sum_val_8')
        print('malloc', sys.getsizeof(sum_val_8), 'sum_val_8')
        for j_8 in range(len(instrument_read(arr_8, 'arr_8'))):
            print(72, 65)
            sum_val_8 += instrument_read_sub(instrument_read(arr_8, 'arr_8'
                ), 'arr_8', instrument_read(j_8, 'j_8'), None, None, False
                ) * instrument_read_sub(instrument_read_sub(instrument_read
                (W_8, 'W_8'), 'W_8', instrument_read(j_8, 'j_8'), None,
                None, False), 'W_8[j_8]', instrument_read(i_8, 'i_8'), None,
                None, False)
            write_instrument_read(sum_val_8, 'sum_val_8')
        print(73, 66)
        result_8[instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
            ] = instrument_read(sum_val_8, 'sum_val_8')
        write_instrument_read_sub(result_8, 'result_8', instrument_read(
            instrument_read(i_8, 'i_8'), 'i_8'), None, None, False)
    print('exit scope 8')
    return instrument_read(result_8, 'result_8')
    print('exit scope 8')


def softmax(arr):
    print('enter scope 9')
    print(1, 69)
    arr_9 = instrument_read(arr, 'arr')
    write_instrument_read(arr_9, 'arr_9')
    print('malloc', sys.getsizeof(arr_9), 'arr_9')
    print(77, 70)
    sum_val_9 = 0
    write_instrument_read(sum_val_9, 'sum_val_9')
    print('malloc', sys.getsizeof(sum_val_9), 'sum_val_9')
    for i_9 in range(len(instrument_read(arr_9, 'arr_9'))):
        print(79, 71)
        sum_val_9 += instrument_read(math, 'math').exp(instrument_read_sub(
            instrument_read(arr_9, 'arr_9'), 'arr_9', instrument_read(i_9,
            'i_9'), None, None, False))
        write_instrument_read(sum_val_9, 'sum_val_9')
    print(80, 72)
    result_9 = instrument_read(np, 'np').zeros(len(instrument_read(arr_9,
        'arr_9')))
    write_instrument_read(result_9, 'result_9')
    print('malloc', sys.getsizeof(result_9), 'result_9')
    for i_9 in range(len(instrument_read(arr_9, 'arr_9'))):
        print(82, 73)
        result_9[instrument_read(instrument_read(i_9, 'i_9'), 'i_9')
            ] = instrument_read(math, 'math').exp(instrument_read_sub(
            instrument_read(arr_9, 'arr_9'), 'arr_9', instrument_read(i_9,
            'i_9'), None, None, False)) / instrument_read(sum_val_9,
            'sum_val_9')
        write_instrument_read_sub(result_9, 'result_9', instrument_read(
            instrument_read(i_9, 'i_9'), 'i_9'), None, None, False)
    print('exit scope 9')
    return instrument_read(result_9, 'result_9')
    print('exit scope 9')


def concat(emb, head, tokens, d_k, cur):
    print('enter scope 10')
    print(1, 76)
    emb_10 = instrument_read(emb, 'emb')
    write_instrument_read(emb_10, 'emb_10')
    print('malloc', sys.getsizeof(emb_10), 'emb_10')
    head_10 = instrument_read(head, 'head')
    write_instrument_read(head_10, 'head_10')
    print('malloc', sys.getsizeof(head_10), 'head_10')
    tokens_10 = instrument_read(tokens, 'tokens')
    write_instrument_read(tokens_10, 'tokens_10')
    print('malloc', sys.getsizeof(tokens_10), 'tokens_10')
    d_k_10 = instrument_read(d_k, 'd_k')
    write_instrument_read(d_k_10, 'd_k_10')
    print('malloc', sys.getsizeof(d_k_10), 'd_k_10')
    cur_10 = instrument_read(cur, 'cur')
    write_instrument_read(cur_10, 'cur_10')
    print('malloc', sys.getsizeof(cur_10), 'cur_10')
    for i_10 in range(instrument_read(tokens_10, 'tokens_10')):
        for j_10 in range(instrument_read(d_k_10, 'd_k_10')):
            print(90, 79)
            emb_10[instrument_read(instrument_read(i_10, 'i_10'), 'i_10')][
                instrument_read(instrument_read(j_10, 'j_10'), 'j_10') + 
                instrument_read(instrument_read(head_10, 'head_10'),
                'head_10') * instrument_read(instrument_read(d_k_10,
                'd_k_10'), 'd_k_10')] = instrument_read_sub(instrument_read_sub
                (instrument_read(cur_10, 'cur_10'), 'cur_10',
                instrument_read(i_10, 'i_10'), None, None, False),
                'cur_10[i_10]', instrument_read(j_10, 'j_10'), None, None,
                False)
            write_instrument_read_sub(emb_10[instrument_read(
                instrument_read(i_10, 'i_10'), 'i_10')],
                "emb_10[instrument_read(i_10, 'i_10')]", instrument_read(
                instrument_read(j_10, 'j_10'), 'j_10') + instrument_read(
                instrument_read(head_10, 'head_10'), 'head_10') *
                instrument_read(instrument_read(d_k_10, 'd_k_10'), 'd_k_10'
                ), None, None, False)
    print('exit scope 10')
    return instrument_read(emb_10, 'emb_10')
    print('exit scope 10')


def self_attn(head, tokens, d_k, Q, K, V):
    print('enter scope 11')
    print(1, 82)
    head_11 = instrument_read(head, 'head')
    write_instrument_read(head_11, 'head_11')
    print('malloc', sys.getsizeof(head_11), 'head_11')
    tokens_11 = instrument_read(tokens, 'tokens')
    write_instrument_read(tokens_11, 'tokens_11')
    print('malloc', sys.getsizeof(tokens_11), 'tokens_11')
    d_k_11 = instrument_read(d_k, 'd_k')
    write_instrument_read(d_k_11, 'd_k_11')
    print('malloc', sys.getsizeof(d_k_11), 'd_k_11')
    Q_11 = instrument_read(Q, 'Q')
    write_instrument_read(Q_11, 'Q_11')
    print('malloc', sys.getsizeof(Q_11), 'Q_11')
    K_11 = instrument_read(K, 'K')
    write_instrument_read(K_11, 'K_11')
    print('malloc', sys.getsizeof(K_11), 'K_11')
    V_11 = instrument_read(V, 'V')
    write_instrument_read(V_11, 'V_11')
    print('malloc', sys.getsizeof(V_11), 'V_11')
    print(95, 83)
    scores_11 = instrument_read(np, 'np').zeros((instrument_read(tokens_11,
        'tokens_11'), instrument_read(tokens_11, 'tokens_11')))
    write_instrument_read(scores_11, 'scores_11')
    print('malloc', sys.getsizeof(scores_11), 'scores_11')
    for i_11 in range(instrument_read(tokens_11, 'tokens_11')):
        for j_11 in range(instrument_read(tokens_11, 'tokens_11')):
            for k_11 in range(instrument_read(d_k_11, 'd_k_11')):
                print(101, 87)
                scores_11[instrument_read(i_11, 'i_11')][instrument_read(
                    j_11, 'j_11')] += instrument_read_sub(instrument_read_sub
                    (instrument_read_sub(instrument_read(Q_11, 'Q_11'),
                    'Q_11', instrument_read(head_11, 'head_11'), None, None,
                    False), 'Q_11[head_11]', instrument_read(i_11, 'i_11'),
                    None, None, False), 'Q_11[head_11][i_11]',
                    instrument_read(k_11, 'k_11'), None, None, False
                    ) * instrument_read_sub(instrument_read_sub(
                    instrument_read_sub(instrument_read(K_11, 'K_11'),
                    'K_11', instrument_read(head_11, 'head_11'), None, None,
                    False), 'K_11[head_11]', instrument_read(j_11, 'j_11'),
                    None, None, False), 'K_11[head_11][j_11]',
                    instrument_read(k_11, 'k_11'), None, None, False)
                write_instrument_read_sub(scores_11[instrument_read(
                    instrument_read(i_11, 'i_11'), 'i_11')],
                    "scores_11[instrument_read(i_11, 'i_11')]",
                    instrument_read(instrument_read(j_11, 'j_11'), 'j_11'),
                    None, None, False)
    for i_11 in range(instrument_read(tokens_11, 'tokens_11')):
        for j_11 in range(instrument_read(tokens_11, 'tokens_11')):
            print(105, 90)
            scores_11[instrument_read(i_11, 'i_11')][instrument_read(j_11,
                'j_11')] /= instrument_read(math, 'math').sqrt(instrument_read
                (d_k_11, 'd_k_11'))
            write_instrument_read_sub(scores_11[instrument_read(
                instrument_read(i_11, 'i_11'), 'i_11')],
                "scores_11[instrument_read(i_11, 'i_11')]", instrument_read
                (instrument_read(j_11, 'j_11'), 'j_11'), None, None, False)
        print(106, 92)
        scores_11 = instrument_read(np, 'np').random.rand(instrument_read(
            tokens_11, 'tokens_11'), instrument_read(tokens_11, 'tokens_11'))
        write_instrument_read(scores_11, 'scores_11')
        print('malloc', sys.getsizeof(scores_11), 'scores_11')
        print(106, 93)
        scores_11[instrument_read(instrument_read(i_11, 'i_11'), 'i_11')
            ] = softmax(instrument_read_sub(instrument_read(scores_11,
            'scores_11'), 'scores_11', instrument_read(i_11, 'i_11'), None,
            None, False))
        write_instrument_read_sub(scores_11, 'scores_11', instrument_read(
            instrument_read(i_11, 'i_11'), 'i_11'), None, None, False)
    print(104, 94)
    out_11 = instrument_read(np, 'np').zeros((instrument_read(tokens_11,
        'tokens_11'), instrument_read(d_k_11, 'd_k_11')))
    write_instrument_read(out_11, 'out_11')
    print('malloc', sys.getsizeof(out_11), 'out_11')
    for i_11 in range(instrument_read(tokens_11, 'tokens_11')):
        for j_11 in range(instrument_read(d_k_11, 'd_k_11')):
            for k_11 in range(instrument_read(tokens_11, 'tokens_11')):
                print(112, 98)
                out_11[instrument_read(i_11, 'i_11')][instrument_read(j_11,
                    'j_11')] += instrument_read_sub(instrument_read_sub(
                    instrument_read(scores_11, 'scores_11'), 'scores_11',
                    instrument_read(i_11, 'i_11'), None, None, False),
                    'scores_11[i_11]', instrument_read(k_11, 'k_11'), None,
                    None, False) * instrument_read_sub(instrument_read_sub(
                    instrument_read_sub(instrument_read(V_11, 'V_11'),
                    'V_11', instrument_read(head_11, 'head_11'), None, None,
                    False), 'V_11[head_11]', instrument_read(k_11, 'k_11'),
                    None, None, False), 'V_11[head_11][k_11]',
                    instrument_read(j_11, 'j_11'), None, None, False)
                write_instrument_read_sub(out_11[instrument_read(
                    instrument_read(i_11, 'i_11'), 'i_11')],
                    "out_11[instrument_read(i_11, 'i_11')]",
                    instrument_read(instrument_read(j_11, 'j_11'), 'j_11'),
                    None, None, False)
    print('exit scope 11')
    return instrument_read(out_11, 'out_11')
    print('exit scope 11')


def main():
    print('enter scope 12')
    print(1, 102)
    print(117, 103)
    d_model_12, heads_12, tokens_12, layers_12 = 100, 12, 8, 12
    write_instrument_read(layers_12, 'layers_12')
    print('malloc', sys.getsizeof(layers_12), 'layers_12')
    print(117, 104)
    d_k_12 = instrument_read(d_model_12, 'd_model_12') // instrument_read(
        heads_12, 'heads_12')
    write_instrument_read(d_k_12, 'd_k_12')
    print('malloc', sys.getsizeof(d_k_12), 'd_k_12')
    print(117, 105)
    embeddings_12 = instrument_read(np, 'np').random.rand(instrument_read(
        tokens_12, 'tokens_12'), instrument_read(d_model_12, 'd_model_12'))
    write_instrument_read(embeddings_12, 'embeddings_12')
    print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
    for i_12 in range(instrument_read(tokens_12, 'tokens_12')):
        for j_12 in range(instrument_read(d_model_12, 'd_model_12')):
            if instrument_read(j_12, 'j_12') % 2 == 0:
                print(123, 109)
                embeddings_12[instrument_read(i_12, 'i_12')][instrument_read
                    (j_12, 'j_12')] += instrument_read(math, 'math').sin(
                    instrument_read(i_12, 'i_12') / instrument_read(math,
                    'math').pow(10000, 2 * instrument_read(j_12, 'j_12') /
                    instrument_read(d_model_12, 'd_model_12')))
                write_instrument_read_sub(embeddings_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')],
                    "embeddings_12[instrument_read(i_12, 'i_12')]",
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12'),
                    None, None, False)
            else:
                print(125, 111)
                embeddings_12[instrument_read(i_12, 'i_12')][instrument_read
                    (j_12, 'j_12')] += instrument_read(math, 'math').cos(
                    instrument_read(i_12, 'i_12') / instrument_read(math,
                    'math').pow(10000, 2 * instrument_read(j_12, 'j_12') /
                    instrument_read(d_model_12, 'd_model_12')))
                write_instrument_read_sub(embeddings_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')],
                    "embeddings_12[instrument_read(i_12, 'i_12')]",
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12'),
                    None, None, False)
    print(120, 112)
    W_Q_12 = balance_random_3d(instrument_read(heads_12, 'heads_12'),
        instrument_read(d_model_12, 'd_model_12'), instrument_read(d_k_12,
        'd_k_12'))
    write_instrument_read(W_Q_12, 'W_Q_12')
    print('malloc', sys.getsizeof(W_Q_12), 'W_Q_12')
    print(120, 113)
    W_K_12 = balance_random_3d(instrument_read(heads_12, 'heads_12'),
        instrument_read(d_model_12, 'd_model_12'), instrument_read(d_k_12,
        'd_k_12'))
    write_instrument_read(W_K_12, 'W_K_12')
    print('malloc', sys.getsizeof(W_K_12), 'W_K_12')
    print(120, 114)
    W_V_12 = balance_random_3d(instrument_read(heads_12, 'heads_12'),
        instrument_read(d_model_12, 'd_model_12'), instrument_read(d_k_12,
        'd_k_12'))
    write_instrument_read(W_V_12, 'W_V_12')
    print('malloc', sys.getsizeof(W_V_12), 'W_V_12')
    print(120, 115)
    Q_12 = instrument_read(np, 'np').zeros((instrument_read(heads_12,
        'heads_12'), instrument_read(tokens_12, 'tokens_12'),
        instrument_read(d_k_12, 'd_k_12')))
    write_instrument_read(Q_12, 'Q_12')
    print('malloc', sys.getsizeof(Q_12), 'Q_12')
    print(120, 116)
    K_12 = instrument_read(np, 'np').zeros((instrument_read(heads_12,
        'heads_12'), instrument_read(tokens_12, 'tokens_12'),
        instrument_read(d_k_12, 'd_k_12')))
    write_instrument_read(K_12, 'K_12')
    print('malloc', sys.getsizeof(K_12), 'K_12')
    print(120, 117)
    V_12 = instrument_read(np, 'np').zeros((instrument_read(heads_12,
        'heads_12'), instrument_read(tokens_12, 'tokens_12'),
        instrument_read(d_k_12, 'd_k_12')))
    write_instrument_read(V_12, 'V_12')
    print('malloc', sys.getsizeof(V_12), 'V_12')
    for i_12 in range(instrument_read(heads_12, 'heads_12')):
        for j_12 in range(instrument_read(tokens_12, 'tokens_12')):
            for k_12 in range(instrument_read(d_k_12, 'd_k_12')):
                print(131, 121)
                sumQ_12, sumK_12, sumV_12 = 0, 0, 0
                write_instrument_read(sumV_12, 'sumV_12')
                print('malloc', sys.getsizeof(sumV_12), 'sumV_12')
                for a_12 in range(instrument_read(d_model_12, 'd_model_12')):
                    print(134, 123)
                    sumQ_12 += instrument_read_sub(instrument_read_sub(
                        instrument_read(embeddings_12, 'embeddings_12'),
                        'embeddings_12', instrument_read(j_12, 'j_12'),
                        None, None, False), 'embeddings_12[j_12]',
                        instrument_read(a_12, 'a_12'), None, None, False
                        ) * instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(W_Q_12,
                        'W_Q_12'), 'W_Q_12', instrument_read(i_12, 'i_12'),
                        None, None, False), 'W_Q_12[i_12]', instrument_read
                        (a_12, 'a_12'), None, None, False),
                        'W_Q_12[i_12][a_12]', instrument_read(k_12, 'k_12'),
                        None, None, False)
                    write_instrument_read(sumQ_12, 'sumQ_12')
                    print(134, 124)
                    sumK_12 += instrument_read_sub(instrument_read_sub(
                        instrument_read(embeddings_12, 'embeddings_12'),
                        'embeddings_12', instrument_read(j_12, 'j_12'),
                        None, None, False), 'embeddings_12[j_12]',
                        instrument_read(a_12, 'a_12'), None, None, False
                        ) * instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(W_K_12,
                        'W_K_12'), 'W_K_12', instrument_read(i_12, 'i_12'),
                        None, None, False), 'W_K_12[i_12]', instrument_read
                        (a_12, 'a_12'), None, None, False),
                        'W_K_12[i_12][a_12]', instrument_read(k_12, 'k_12'),
                        None, None, False)
                    write_instrument_read(sumK_12, 'sumK_12')
                    print(134, 125)
                    sumV_12 += instrument_read_sub(instrument_read_sub(
                        instrument_read(embeddings_12, 'embeddings_12'),
                        'embeddings_12', instrument_read(j_12, 'j_12'),
                        None, None, False), 'embeddings_12[j_12]',
                        instrument_read(a_12, 'a_12'), None, None, False
                        ) * instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(W_V_12,
                        'W_V_12'), 'W_V_12', instrument_read(i_12, 'i_12'),
                        None, None, False), 'W_V_12[i_12]', instrument_read
                        (a_12, 'a_12'), None, None, False),
                        'W_V_12[i_12][a_12]', instrument_read(k_12, 'k_12'),
                        None, None, False)
                    write_instrument_read(sumV_12, 'sumV_12')
                print(135, 126)
                Q_12[instrument_read(instrument_read(i_12, 'i_12'), 'i_12')][
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12')][
                    instrument_read(instrument_read(k_12, 'k_12'), 'k_12')
                    ] = instrument_read(sumQ_12, 'sumQ_12')
                write_instrument_read_sub(Q_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')][instrument_read
                    (instrument_read(j_12, 'j_12'), 'j_12')],
                    "Q_12[instrument_read(i_12, 'i_12')][instrument_read(j_12, 'j_12')]"
                    , instrument_read(instrument_read(k_12, 'k_12'), 'k_12'
                    ), None, None, False)
                print(135, 127)
                K_12[instrument_read(instrument_read(i_12, 'i_12'), 'i_12')][
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12')][
                    instrument_read(instrument_read(k_12, 'k_12'), 'k_12')
                    ] = instrument_read(sumK_12, 'sumK_12')
                write_instrument_read_sub(K_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')][instrument_read
                    (instrument_read(j_12, 'j_12'), 'j_12')],
                    "K_12[instrument_read(i_12, 'i_12')][instrument_read(j_12, 'j_12')]"
                    , instrument_read(instrument_read(k_12, 'k_12'), 'k_12'
                    ), None, None, False)
                print(135, 128)
                V_12[instrument_read(instrument_read(i_12, 'i_12'), 'i_12')][
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12')][
                    instrument_read(instrument_read(k_12, 'k_12'), 'k_12')
                    ] = instrument_read(sumV_12, 'sumV_12')
                write_instrument_read_sub(V_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')][instrument_read
                    (instrument_read(j_12, 'j_12'), 'j_12')],
                    "V_12[instrument_read(i_12, 'i_12')][instrument_read(j_12, 'j_12')]"
                    , instrument_read(instrument_read(k_12, 'k_12'), 'k_12'
                    ), None, None, False)
    for i_12 in range(instrument_read(layers_12, 'layers_12')):
        print(136, 130)
        emb_cpy_12 = instrument_read(np, 'np').copy(instrument_read(
            embeddings_12, 'embeddings_12'))
        write_instrument_read(emb_cpy_12, 'emb_cpy_12')
        print('malloc', sys.getsizeof(emb_cpy_12), 'emb_cpy_12')
        print(136, 131)
        multi_head_out_12 = instrument_read(np, 'np').zeros((
            instrument_read(tokens_12, 'tokens_12'), instrument_read(
            d_model_12, 'd_model_12')))
        write_instrument_read(multi_head_out_12, 'multi_head_out_12')
        print('malloc', sys.getsizeof(multi_head_out_12), 'multi_head_out_12')
        for j_12 in range(instrument_read(heads_12, 'heads_12')):
            print(139, 133)
            cur_12 = self_attn(instrument_read(j_12, 'j_12'),
                instrument_read(tokens_12, 'tokens_12'), instrument_read(
                d_k_12, 'd_k_12'), instrument_read(Q_12, 'Q_12'),
                instrument_read(K_12, 'K_12'), instrument_read(V_12, 'V_12'))
            write_instrument_read(cur_12, 'cur_12')
            print('malloc', sys.getsizeof(cur_12), 'cur_12')
            print(139, 134)
            multi_head_out_12 = concat(instrument_read(multi_head_out_12,
                'multi_head_out_12'), instrument_read(j_12, 'j_12'),
                instrument_read(tokens_12, 'tokens_12'), instrument_read(
                d_k_12, 'd_k_12'), instrument_read(cur_12, 'cur_12'))
            write_instrument_read(multi_head_out_12, 'multi_head_out_12')
            print('malloc', sys.getsizeof(multi_head_out_12),
                'multi_head_out_12')
        print(140, 135)
        W_attn_12 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model_12, 'd_model_12'), instrument_read(d_model_12,
            'd_model_12'))
        write_instrument_read(W_attn_12, 'W_attn_12')
        print('malloc', sys.getsizeof(W_attn_12), 'W_attn_12')
        for i_12 in range(instrument_read(tokens_12, 'tokens_12')):
            for j_12 in range(instrument_read(d_model_12, 'd_model_12')):
                print(144, 138)
                sum_val_12 = 0
                write_instrument_read(sum_val_12, 'sum_val_12')
                print('malloc', sys.getsizeof(sum_val_12), 'sum_val_12')
                for k_12 in range(instrument_read(d_model_12, 'd_model_12')):
                    print(147, 140)
                    sum_val_12 += instrument_read_sub(instrument_read_sub(
                        instrument_read(multi_head_out_12,
                        'multi_head_out_12'), 'multi_head_out_12',
                        instrument_read(i_12, 'i_12'), None, None, False),
                        'multi_head_out_12[i_12]', instrument_read(k_12,
                        'k_12'), None, None, False) * instrument_read_sub(
                        instrument_read_sub(instrument_read(W_attn_12,
                        'W_attn_12'), 'W_attn_12', instrument_read(k_12,
                        'k_12'), None, None, False), 'W_attn_12[k_12]',
                        instrument_read(j_12, 'j_12'), None, None, False)
                    write_instrument_read(sum_val_12, 'sum_val_12')
                print(148, 141)
                embeddings_12[instrument_read(instrument_read(i_12, 'i_12'),
                    'i_12')][instrument_read(instrument_read(j_12, 'j_12'),
                    'j_12')] = instrument_read(sum_val_12, 'sum_val_12')
                write_instrument_read_sub(embeddings_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')],
                    "embeddings_12[instrument_read(i_12, 'i_12')]",
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12'),
                    None, None, False)
        print(143, 142)
        embeddings_12 = arr_add(instrument_read(embeddings_12,
            'embeddings_12'), instrument_read(emb_cpy_12, 'emb_cpy_12'))
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
        print(143, 143)
        weights_12, biases_12 = instrument_read(np, 'np').random.rand(
            instrument_read(d_model_12, 'd_model_12')), instrument_read(np,
            'np').random.rand(instrument_read(d_model_12, 'd_model_12'))
        write_instrument_read(biases_12, 'biases_12')
        print('malloc', sys.getsizeof(biases_12), 'biases_12')
        print(143, 144)
        embeddings_12 = BN_layer(instrument_read(embeddings_12,
            'embeddings_12'), instrument_read(weights_12, 'weights_12'),
            instrument_read(biases_12, 'biases_12'))
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
        print(143, 145)
        emb_cpy_12 = instrument_read(np, 'np').copy(instrument_read(
            embeddings_12, 'embeddings_12'))
        write_instrument_read(emb_cpy_12, 'emb_cpy_12')
        print('malloc', sys.getsizeof(emb_cpy_12), 'emb_cpy_12')
        print(143, 146)
        W_12 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model_12, 'd_model_12'), instrument_read(d_model_12,
            'd_model_12') * 4)
        write_instrument_read(W_12, 'W_12')
        print('malloc', sys.getsizeof(W_12), 'W_12')
        print(143, 147)
        W_0_12 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model_12, 'd_model_12') * 4)
        write_instrument_read(W_0_12, 'W_0_12')
        print('malloc', sys.getsizeof(W_0_12), 'W_0_12')
        print(143, 148)
        emb_new_12 = instrument_read(np, 'np').zeros((instrument_read(
            tokens_12, 'tokens_12'), instrument_read(d_model_12,
            'd_model_12') * 4))
        write_instrument_read(emb_new_12, 'emb_new_12')
        print('malloc', sys.getsizeof(emb_new_12), 'emb_new_12')
        for i_12 in range(instrument_read(tokens_12, 'tokens_12')):
            print(150, 150)
            emb_new_12[instrument_read(instrument_read(i_12, 'i_12'), 'i_12')
                ] = fc_layer(instrument_read_sub(instrument_read(
                embeddings_12, 'embeddings_12'), 'embeddings_12',
                instrument_read(i_12, 'i_12'), None, None, False),
                instrument_read(W_12, 'W_12'), instrument_read(W_0_12,
                'W_0_12'))
            write_instrument_read_sub(emb_new_12, 'emb_new_12',
                instrument_read(instrument_read(i_12, 'i_12'), 'i_12'),
                None, None, False)
        print(151, 151)
        embeddings_12 = instrument_read(emb_new_12, 'emb_new_12')
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
        print(151, 152)
        embeddings_12 = reLU(instrument_read(embeddings_12, 'embeddings_12'))
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
        print(151, 153)
        W_12 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model_12, 'd_model_12') * 4, instrument_read(d_model_12,
            'd_model_12'))
        write_instrument_read(W_12, 'W_12')
        print('malloc', sys.getsizeof(W_12), 'W_12')
        print(151, 154)
        W_0_12 = instrument_read(np, 'np').random.rand(instrument_read(
            d_model_12, 'd_model_12'))
        write_instrument_read(W_0_12, 'W_0_12')
        print('malloc', sys.getsizeof(W_0_12), 'W_0_12')
        print(151, 155)
        emb_new_12 = instrument_read(np, 'np').zeros((instrument_read(
            tokens_12, 'tokens_12'), instrument_read(d_model_12, 'd_model_12'))
            )
        write_instrument_read(emb_new_12, 'emb_new_12')
        print('malloc', sys.getsizeof(emb_new_12), 'emb_new_12')
        for i_12 in range(instrument_read(tokens_12, 'tokens_12')):
            print(153, 157)
            emb_new_12[instrument_read(instrument_read(i_12, 'i_12'), 'i_12')
                ] = fc_layer(instrument_read_sub(instrument_read(
                embeddings_12, 'embeddings_12'), 'embeddings_12',
                instrument_read(i_12, 'i_12'), None, None, False),
                instrument_read(W_12, 'W_12'), instrument_read(W_0_12,
                'W_0_12'))
            write_instrument_read_sub(emb_new_12, 'emb_new_12',
                instrument_read(instrument_read(i_12, 'i_12'), 'i_12'),
                None, None, False)
        print(154, 158)
        embeddings_12 = instrument_read(emb_new_12, 'emb_new_12')
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
        print(154, 159)
        embeddings_12 = arr_add(instrument_read(embeddings_12,
            'embeddings_12'), instrument_read(emb_cpy_12, 'emb_cpy_12'))
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
        print(154, 160)
        embeddings_12 = BN_layer(instrument_read(embeddings_12,
            'embeddings_12'), instrument_read(weights_12, 'weights_12'),
            instrument_read(biases_12, 'biases_12'))
        write_instrument_read(embeddings_12, 'embeddings_12')
        print('malloc', sys.getsizeof(embeddings_12), 'embeddings_12')
    print('exit scope 12')


if instrument_read(__name__, '__name__') == '__main__':
    instrument_read(loop, 'loop').start_unroll
    main()
