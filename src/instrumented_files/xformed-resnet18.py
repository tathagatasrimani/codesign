import sys
from instrument_lib import *
import sys
from instrument_lib import *
import math
import numpy as np
from loop import loop


def zero_pad_arr(img, zero_pad):
    print('enter scope 1')
    print(1, 5)
    img_1 = instrument_read(img, 'img')
    write_instrument_read(img_1, 'img_1')
    print('malloc', sys.getsizeof(img_1), 'img_1')
    zero_pad_1 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad_1, 'zero_pad_1')
    print('malloc', sys.getsizeof(zero_pad_1), 'zero_pad_1')
    print(3, 6)
    len_new_1 = len(instrument_read_sub(instrument_read(img_1, 'img_1'),
        'img_1', 0, None, None, False)) + 2 * instrument_read(zero_pad_1,
        'zero_pad_1')
    write_instrument_read(len_new_1, 'len_new_1')
    print('malloc', sys.getsizeof(len_new_1), 'len_new_1')
    print(3, 7)
    wid_new_1 = len(instrument_read_sub(instrument_read_sub(instrument_read
        (img_1, 'img_1'), 'img_1', 0, None, None, False), 'img_1[0]', 0,
        None, None, False)) + 2 * instrument_read(zero_pad_1, 'zero_pad_1')
    write_instrument_read(wid_new_1, 'wid_new_1')
    print('malloc', sys.getsizeof(wid_new_1), 'wid_new_1')
    print(3, 8)
    new_img_1 = instrument_read(np, 'np').zeros((len(instrument_read(img_1,
        'img_1')), instrument_read(len_new_1, 'len_new_1'), instrument_read
        (wid_new_1, 'wid_new_1')))
    write_instrument_read(new_img_1, 'new_img_1')
    print('malloc', sys.getsizeof(new_img_1), 'new_img_1')
    for i_1 in range(len(instrument_read(img_1, 'img_1'))):
        for j_1 in range(instrument_read(len_new_1, 'len_new_1')):
            print(7, 11)
            make_zero_1 = instrument_read(j_1, 'j_1') < instrument_read(
                zero_pad_1, 'zero_pad_1') or instrument_read(j_1, 'j_1'
                ) >= instrument_read(len_new_1, 'len_new_1') - instrument_read(
                zero_pad_1, 'zero_pad_1')
            write_instrument_read(make_zero_1, 'make_zero_1')
            print('malloc', sys.getsizeof(make_zero_1), 'make_zero_1')
            for k_1 in range(instrument_read(wid_new_1, 'wid_new_1')):
                if instrument_read(k_1, 'k_1') < instrument_read(zero_pad_1,
                    'zero_pad_1') or instrument_read(k_1, 'k_1'
                    ) >= instrument_read(wid_new_1, 'wid_new_1'
                    ) - instrument_read(zero_pad_1, 'zero_pad_1'
                    ) or instrument_read(make_zero_1, 'make_zero_1'):
                    print(12, 14)
                    new_img_1[instrument_read(instrument_read(i_1, 'i_1'),
                        'i_1')][instrument_read(instrument_read(j_1, 'j_1'),
                        'j_1')][instrument_read(instrument_read(k_1, 'k_1'),
                        'k_1')] = 0
                    write_instrument_read_sub(new_img_1[instrument_read(
                        instrument_read(i_1, 'i_1'), 'i_1')][
                        instrument_read(instrument_read(j_1, 'j_1'), 'j_1')
                        ],
                        "new_img_1[instrument_read(i_1, 'i_1')][instrument_read(j_1, 'j_1')]"
                        , instrument_read(instrument_read(k_1, 'k_1'),
                        'k_1'), None, None, False)
                else:
                    print(14, 16)
                    new_img_1[instrument_read(instrument_read(i_1, 'i_1'),
                        'i_1')][instrument_read(instrument_read(j_1, 'j_1'),
                        'j_1')][instrument_read(instrument_read(k_1, 'k_1'),
                        'k_1')] = instrument_read_sub(instrument_read_sub(
                        instrument_read_sub(instrument_read(img_1, 'img_1'),
                        'img_1', instrument_read(i_1, 'i_1'), None, None,
                        False), 'img_1[i_1]', instrument_read(j_1, 'j_1') -
                        instrument_read(zero_pad_1, 'zero_pad_1'), None,
                        None, False), 'img_1[i_1][j_1 - zero_pad_1]', 
                        instrument_read(k_1, 'k_1') - instrument_read(
                        zero_pad_1, 'zero_pad_1'), None, None, False)
                    write_instrument_read_sub(new_img_1[instrument_read(
                        instrument_read(i_1, 'i_1'), 'i_1')][
                        instrument_read(instrument_read(j_1, 'j_1'), 'j_1')
                        ],
                        "new_img_1[instrument_read(i_1, 'i_1')][instrument_read(j_1, 'j_1')]"
                        , instrument_read(instrument_read(k_1, 'k_1'),
                        'k_1'), None, None, False)
    print('exit scope 1')
    return instrument_read(new_img_1, 'new_img_1')
    print('exit scope 1')


def conv_layer(img, filt, numFilt, zero_pad, stride):
    print('enter scope 2')
    print(1, 20)
    img_2 = instrument_read(img, 'img')
    write_instrument_read(img_2, 'img_2')
    print('malloc', sys.getsizeof(img_2), 'img_2')
    filt_2 = instrument_read(filt, 'filt')
    write_instrument_read(filt_2, 'filt_2')
    print('malloc', sys.getsizeof(filt_2), 'filt_2')
    numFilt_2 = instrument_read(numFilt, 'numFilt')
    write_instrument_read(numFilt_2, 'numFilt_2')
    print('malloc', sys.getsizeof(numFilt_2), 'numFilt_2')
    zero_pad_2 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad_2, 'zero_pad_2')
    print('malloc', sys.getsizeof(zero_pad_2), 'zero_pad_2')
    stride_2 = instrument_read(stride, 'stride')
    write_instrument_read(stride_2, 'stride_2')
    print('malloc', sys.getsizeof(stride_2), 'stride_2')
    print(18, 21)
    f_len_2 = int((len(instrument_read_sub(instrument_read(img_2, 'img_2'),
        'img_2', 0, None, None, False)) - len(instrument_read_sub(
        instrument_read(filt_2, 'filt_2'), 'filt_2', 0, None, None, False)) +
        2 * instrument_read(zero_pad_2, 'zero_pad_2')) / instrument_read(
        stride_2, 'stride_2') + 1)
    write_instrument_read(f_len_2, 'f_len_2')
    print('malloc', sys.getsizeof(f_len_2), 'f_len_2')
    print(18, 22)
    f_wid_2 = int((len(instrument_read_sub(instrument_read_sub(
        instrument_read(img_2, 'img_2'), 'img_2', 0, None, None, False),
        'img_2[0]', 0, None, None, False)) - len(instrument_read_sub(
        instrument_read_sub(instrument_read(filt_2, 'filt_2'), 'filt_2', 0,
        None, None, False), 'filt_2[0]', 0, None, None, False)) + 2 *
        instrument_read(zero_pad_2, 'zero_pad_2')) / instrument_read(
        stride_2, 'stride_2') + 1)
    write_instrument_read(f_wid_2, 'f_wid_2')
    print('malloc', sys.getsizeof(f_wid_2), 'f_wid_2')
    print(18, 23)
    biases_2 = instrument_read(np, 'np').random.rand(instrument_read(
        f_len_2, 'f_len_2'))
    write_instrument_read(biases_2, 'biases_2')
    print('malloc', sys.getsizeof(biases_2), 'biases_2')
    print(18, 24)
    img_new_2 = zero_pad_arr(instrument_read(img_2, 'img_2'),
        instrument_read(zero_pad_2, 'zero_pad_2'))
    write_instrument_read(img_new_2, 'img_new_2')
    print('malloc', sys.getsizeof(img_new_2), 'img_new_2')
    print(18, 25)
    f_new_2 = instrument_read(np, 'np').zeros((instrument_read(numFilt_2,
        'numFilt_2'), instrument_read(f_len_2, 'f_len_2'), instrument_read(
        f_wid_2, 'f_wid_2')))
    write_instrument_read(f_new_2, 'f_new_2')
    print('malloc', sys.getsizeof(f_new_2), 'f_new_2')
    for i_2 in range(instrument_read(numFilt_2, 'numFilt_2')):
        for j_2 in range(instrument_read(f_len_2, 'f_len_2')):
            for k_2 in range(instrument_read(f_wid_2, 'f_wid_2')):
                for l_2 in range(len(instrument_read(filt_2, 'filt_2'))):
                    for c_2 in range(len(instrument_read_sub(
                        instrument_read(filt_2, 'filt_2'), 'filt_2', 0,
                        None, None, False))):
                        for d_2 in range(len(instrument_read_sub(
                            instrument_read_sub(instrument_read(filt_2,
                            'filt_2'), 'filt_2', 0, None, None, False),
                            'filt_2[0]', 0, None, None, False))):
                            print(30, 32)
                            f_new_2[instrument_read(i_2, 'i_2')][
                                instrument_read(j_2, 'j_2')][instrument_read
                                (k_2, 'k_2')] += instrument_read_sub(
                                instrument_read_sub(instrument_read_sub(
                                instrument_read(img_new_2, 'img_new_2'),
                                'img_new_2', instrument_read(l_2, 'l_2'),
                                None, None, False), 'img_new_2[l_2]', 
                                instrument_read(j_2, 'j_2') *
                                instrument_read(stride_2, 'stride_2') +
                                instrument_read(c_2, 'c_2'), None, None,
                                False),
                                'img_new_2[l_2][j_2 * stride_2 + c_2]', 
                                instrument_read(k_2, 'k_2') *
                                instrument_read(stride_2, 'stride_2') +
                                instrument_read(d_2, 'd_2'), None, None, False
                                ) * instrument_read_sub(instrument_read_sub
                                (instrument_read_sub(instrument_read(filt_2,
                                'filt_2'), 'filt_2', instrument_read(l_2,
                                'l_2'), None, None, False), 'filt_2[l_2]',
                                instrument_read(c_2, 'c_2'), None, None,
                                False), 'filt_2[l_2][c_2]', instrument_read
                                (d_2, 'd_2'), None, None, False)
                            write_instrument_read_sub(f_new_2[
                                instrument_read(instrument_read(i_2, 'i_2'),
                                'i_2')][instrument_read(instrument_read(j_2,
                                'j_2'), 'j_2')],
                                "f_new_2[instrument_read(i_2, 'i_2')][instrument_read(j_2, 'j_2')]"
                                , instrument_read(instrument_read(k_2,
                                'k_2'), 'k_2'), None, None, False)
    for i_2 in range(instrument_read(numFilt_2, 'numFilt_2')):
        for j_2 in range(instrument_read(f_len_2, 'f_len_2')):
            for k_2 in range(instrument_read(f_wid_2, 'f_wid_2')):
                print(36, 36)
                f_new_2[instrument_read(i_2, 'i_2')][instrument_read(j_2,
                    'j_2')][instrument_read(k_2, 'k_2')
                    ] += instrument_read_sub(instrument_read(biases_2,
                    'biases_2'), 'biases_2', instrument_read(j_2, 'j_2'),
                    None, None, False)
                write_instrument_read_sub(f_new_2[instrument_read(
                    instrument_read(i_2, 'i_2'), 'i_2')][instrument_read(
                    instrument_read(j_2, 'j_2'), 'j_2')],
                    "f_new_2[instrument_read(i_2, 'i_2')][instrument_read(j_2, 'j_2')]"
                    , instrument_read(instrument_read(k_2, 'k_2'), 'k_2'),
                    None, None, False)
    print('exit scope 2')
    return instrument_read(f_new_2, 'f_new_2')
    print('exit scope 2')


def max_pool(input, l, w, zero_pad, stride):
    print('enter scope 3')
    print(1, 39)
    input_3 = instrument_read(input, 'input')
    write_instrument_read(input_3, 'input_3')
    print('malloc', sys.getsizeof(input_3), 'input_3')
    l_3 = instrument_read(l, 'l')
    write_instrument_read(l_3, 'l_3')
    print('malloc', sys.getsizeof(l_3), 'l_3')
    w_3 = instrument_read(w, 'w')
    write_instrument_read(w_3, 'w_3')
    print('malloc', sys.getsizeof(w_3), 'w_3')
    zero_pad_3 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad_3, 'zero_pad_3')
    print('malloc', sys.getsizeof(zero_pad_3), 'zero_pad_3')
    stride_3 = instrument_read(stride, 'stride')
    write_instrument_read(stride_3, 'stride_3')
    print('malloc', sys.getsizeof(stride_3), 'stride_3')
    if instrument_read(zero_pad_3, 'zero_pad_3') > 0:
        print(42, 41)
        input_3 = zero_pad_arr(instrument_read(input_3, 'input_3'),
            instrument_read(zero_pad_3, 'zero_pad_3'))
        write_instrument_read(input_3, 'input_3')
        print('malloc', sys.getsizeof(input_3), 'input_3')
    print(43, 42)
    res_l_3 = int((len(instrument_read_sub(instrument_read(input_3,
        'input_3'), 'input_3', 0, None, None, False)) - instrument_read(l_3,
        'l_3')) / instrument_read(stride_3, 'stride_3') + 1)
    write_instrument_read(res_l_3, 'res_l_3')
    print('malloc', sys.getsizeof(res_l_3), 'res_l_3')
    print(43, 43)
    res_w_3 = int((len(instrument_read_sub(instrument_read_sub(
        instrument_read(input_3, 'input_3'), 'input_3', 0, None, None,
        False), 'input_3[0]', 0, None, None, False)) - instrument_read(w_3,
        'w_3')) / instrument_read(stride_3, 'stride_3') + 1)
    write_instrument_read(res_w_3, 'res_w_3')
    print('malloc', sys.getsizeof(res_w_3), 'res_w_3')
    print(43, 44)
    result_3 = instrument_read(np, 'np').zeros((len(instrument_read(input_3,
        'input_3')), instrument_read(res_l_3, 'res_l_3'), instrument_read(
        res_w_3, 'res_w_3')))
    write_instrument_read(result_3, 'result_3')
    print('malloc', sys.getsizeof(result_3), 'result_3')
    for i_3 in range(len(instrument_read(input_3, 'input_3'))):
        for j_3 in range(instrument_read(res_l_3, 'res_l_3')):
            for k_3 in range(instrument_read(res_w_3, 'res_w_3')):
                for c_3 in range(instrument_read(l_3, 'l_3')):
                    for d_3 in range(instrument_read(w_3, 'w_3')):
                        print(53, 50)
                        result_3[instrument_read(instrument_read(i_3, 'i_3'
                            ), 'i_3')][instrument_read(instrument_read(j_3,
                            'j_3'), 'j_3')][instrument_read(instrument_read
                            (k_3, 'k_3'), 'k_3')] = max(instrument_read_sub
                            (instrument_read_sub(instrument_read_sub(
                            instrument_read(input_3, 'input_3'), 'input_3',
                            instrument_read(i_3, 'i_3'), None, None, False),
                            'input_3[i_3]', instrument_read(j_3, 'j_3'),
                            None, None, False), 'input_3[i_3][j_3]',
                            instrument_read(k_3, 'k_3'), None, None, False),
                            instrument_read_sub(instrument_read_sub(
                            instrument_read_sub(instrument_read(result_3,
                            'result_3'), 'result_3', instrument_read(i_3,
                            'i_3'), None, None, False), 'result_3[i_3]',
                            instrument_read(j_3, 'j_3'), None, None, False),
                            'result_3[i_3][j_3]', instrument_read(k_3,
                            'k_3'), None, None, False), instrument_read_sub
                            (instrument_read_sub(instrument_read_sub(
                            instrument_read(input_3, 'input_3'), 'input_3',
                            instrument_read(i_3, 'i_3'), None, None, False),
                            'input_3[i_3]', instrument_read(j_3, 'j_3') *
                            instrument_read(stride_3, 'stride_3') +
                            instrument_read(c_3, 'c_3'), None, None, False),
                            'input_3[i_3][j_3 * stride_3 + c_3]', 
                            instrument_read(k_3, 'k_3') * instrument_read(
                            stride_3, 'stride_3') + instrument_read(d_3,
                            'd_3'), None, None, False))
                        write_instrument_read_sub(result_3[instrument_read(
                            instrument_read(i_3, 'i_3'), 'i_3')][
                            instrument_read(instrument_read(j_3, 'j_3'),
                            'j_3')],
                            "result_3[instrument_read(i_3, 'i_3')][instrument_read(j_3, 'j_3')]"
                            , instrument_read(instrument_read(k_3, 'k_3'),
                            'k_3'), None, None, False)
    for i_3 in range(len(instrument_read(input_3, 'input_3'))):
        for j_3 in range(instrument_read(res_l_3, 'res_l_3')):
            for k_3 in range(instrument_read(res_w_3, 'res_w_3')):
                print(59, 54)
                result_3[instrument_read(i_3, 'i_3')][instrument_read(j_3,
                    'j_3')][instrument_read(k_3, 'k_3')] /= instrument_read(l_3
                    , 'l_3') * instrument_read(w_3, 'w_3')
                write_instrument_read_sub(result_3[instrument_read(
                    instrument_read(i_3, 'i_3'), 'i_3')][instrument_read(
                    instrument_read(j_3, 'j_3'), 'j_3')],
                    "result_3[instrument_read(i_3, 'i_3')][instrument_read(j_3, 'j_3')]"
                    , instrument_read(instrument_read(k_3, 'k_3'), 'k_3'),
                    None, None, False)
    print('exit scope 3')
    return instrument_read(result_3, 'result_3')
    print('exit scope 3')


def avg_pool(input, l, w, zero_pad, stride):
    print('enter scope 4')
    print(1, 57)
    input_4 = instrument_read(input, 'input')
    write_instrument_read(input_4, 'input_4')
    print('malloc', sys.getsizeof(input_4), 'input_4')
    l_4 = instrument_read(l, 'l')
    write_instrument_read(l_4, 'l_4')
    print('malloc', sys.getsizeof(l_4), 'l_4')
    w_4 = instrument_read(w, 'w')
    write_instrument_read(w_4, 'w_4')
    print('malloc', sys.getsizeof(w_4), 'w_4')
    zero_pad_4 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad_4, 'zero_pad_4')
    print('malloc', sys.getsizeof(zero_pad_4), 'zero_pad_4')
    stride_4 = instrument_read(stride, 'stride')
    write_instrument_read(stride_4, 'stride_4')
    print('malloc', sys.getsizeof(stride_4), 'stride_4')
    if instrument_read(zero_pad_4, 'zero_pad_4') > 0:
        print(65, 59)
        input_4 = zero_pad_arr(instrument_read(input_4, 'input_4'),
            instrument_read(zero_pad_4, 'zero_pad_4'))
        write_instrument_read(input_4, 'input_4')
        print('malloc', sys.getsizeof(input_4), 'input_4')
    print(66, 60)
    res_l_4 = int((len(instrument_read_sub(instrument_read(input_4,
        'input_4'), 'input_4', 0, None, None, False)) - instrument_read(l_4,
        'l_4')) / instrument_read(stride_4, 'stride_4') + 1)
    write_instrument_read(res_l_4, 'res_l_4')
    print('malloc', sys.getsizeof(res_l_4), 'res_l_4')
    print(66, 61)
    res_w_4 = int((len(instrument_read_sub(instrument_read_sub(
        instrument_read(input_4, 'input_4'), 'input_4', 0, None, None,
        False), 'input_4[0]', 0, None, None, False)) - instrument_read(w_4,
        'w_4')) / instrument_read(stride_4, 'stride_4') + 1)
    write_instrument_read(res_w_4, 'res_w_4')
    print('malloc', sys.getsizeof(res_w_4), 'res_w_4')
    print(66, 62)
    result_4 = instrument_read(np, 'np').zeros((len(instrument_read(input_4,
        'input_4')), instrument_read(res_l_4, 'res_l_4'), instrument_read(
        res_w_4, 'res_w_4')))
    write_instrument_read(result_4, 'result_4')
    print('malloc', sys.getsizeof(result_4), 'result_4')
    for i_4 in range(len(instrument_read(input_4, 'input_4'))):
        for j_4 in range(instrument_read(res_l_4, 'res_l_4')):
            for k_4 in range(instrument_read(res_w_4, 'res_w_4')):
                for c_4 in range(instrument_read(l_4, 'l_4')):
                    for d_4 in range(instrument_read(w_4, 'w_4')):
                        print(76, 68)
                        result_4[instrument_read(i_4, 'i_4')][instrument_read
                            (j_4, 'j_4')][instrument_read(k_4, 'k_4')
                            ] += instrument_read_sub(instrument_read_sub(
                            instrument_read_sub(instrument_read(input_4,
                            'input_4'), 'input_4', instrument_read(i_4,
                            'i_4'), None, None, False), 'input_4[i_4]', 
                            instrument_read(j_4, 'j_4') * instrument_read(
                            stride_4, 'stride_4') + instrument_read(c_4,
                            'c_4'), None, None, False),
                            'input_4[i_4][j_4 * stride_4 + c_4]', 
                            instrument_read(k_4, 'k_4') * instrument_read(
                            stride_4, 'stride_4') + instrument_read(d_4,
                            'd_4'), None, None, False)
                        write_instrument_read_sub(result_4[instrument_read(
                            instrument_read(i_4, 'i_4'), 'i_4')][
                            instrument_read(instrument_read(j_4, 'j_4'),
                            'j_4')],
                            "result_4[instrument_read(i_4, 'i_4')][instrument_read(j_4, 'j_4')]"
                            , instrument_read(instrument_read(k_4, 'k_4'),
                            'k_4'), None, None, False)
    for i_4 in range(len(instrument_read(input_4, 'input_4'))):
        for j_4 in range(instrument_read(res_l_4, 'res_l_4')):
            for k_4 in range(instrument_read(res_w_4, 'res_w_4')):
                print(82, 72)
                result_4[instrument_read(i_4, 'i_4')][instrument_read(j_4,
                    'j_4')][instrument_read(k_4, 'k_4')] /= instrument_read(l_4
                    , 'l_4') * instrument_read(w_4, 'w_4')
                write_instrument_read_sub(result_4[instrument_read(
                    instrument_read(i_4, 'i_4'), 'i_4')][instrument_read(
                    instrument_read(j_4, 'j_4'), 'j_4')],
                    "result_4[instrument_read(i_4, 'i_4')][instrument_read(j_4, 'j_4')]"
                    , instrument_read(instrument_read(k_4, 'k_4'), 'k_4'),
                    None, None, False)
    print('exit scope 4')
    return instrument_read(result_4, 'result_4')
    print('exit scope 4')


def reLU(img):
    print('enter scope 5')
    print(1, 75)
    img_5 = instrument_read(img, 'img')
    write_instrument_read(img_5, 'img_5')
    print('malloc', sys.getsizeof(img_5), 'img_5')
    for i_5 in range(len(instrument_read(img_5, 'img_5'))):
        for j_5 in range(len(instrument_read_sub(instrument_read(img_5,
            'img_5'), 'img_5', 0, None, None, False))):
            for k_5 in range(len(instrument_read_sub(instrument_read_sub(
                instrument_read(img_5, 'img_5'), 'img_5', 0, None, None,
                False), 'img_5[0]', 0, None, None, False))):
                print(92, 79)
                img_5[instrument_read(instrument_read(i_5, 'i_5'), 'i_5')][
                    instrument_read(instrument_read(j_5, 'j_5'), 'j_5')][
                    instrument_read(instrument_read(k_5, 'k_5'), 'k_5')] = max(
                    instrument_read_sub(instrument_read_sub(
                    instrument_read_sub(instrument_read(img_5, 'img_5'),
                    'img_5', instrument_read(i_5, 'i_5'), None, None, False
                    ), 'img_5[i_5]', instrument_read(j_5, 'j_5'), None,
                    None, False), 'img_5[i_5][j_5]', instrument_read(k_5,
                    'k_5'), None, None, False), 0)
                write_instrument_read_sub(img_5[instrument_read(
                    instrument_read(i_5, 'i_5'), 'i_5')][instrument_read(
                    instrument_read(j_5, 'j_5'), 'j_5')],
                    "img_5[instrument_read(i_5, 'i_5')][instrument_read(j_5, 'j_5')]"
                    , instrument_read(instrument_read(k_5, 'k_5'), 'k_5'),
                    None, None, False)
    print('exit scope 5')
    return instrument_read(img_5, 'img_5')
    print('exit scope 5')


def get_mean(row):
    print('enter scope 6')
    print(1, 82)
    row_6 = instrument_read(row, 'row')
    write_instrument_read(row_6, 'row_6')
    print('malloc', sys.getsizeof(row_6), 'row_6')
    print(97, 83)
    sum_val_6 = 0
    write_instrument_read(sum_val_6, 'sum_val_6')
    print('malloc', sys.getsizeof(sum_val_6), 'sum_val_6')
    for i_6 in range(len(instrument_read(row_6, 'row_6'))):
        print(99, 85)
        sum_val_6 += instrument_read_sub(instrument_read(row_6, 'row_6'),
            'row_6', instrument_read(i_6, 'i_6'), None, None, False)
        write_instrument_read(sum_val_6, 'sum_val_6')
    print('exit scope 6')
    return instrument_read(sum_val_6, 'sum_val_6') / len(instrument_read(
        row_6, 'row_6'))
    print('exit scope 6')


def std_dev(row):
    print('enter scope 7')
    print(1, 88)
    row_7 = instrument_read(row, 'row')
    write_instrument_read(row_7, 'row_7')
    print('malloc', sys.getsizeof(row_7), 'row_7')
    print(104, 89)
    result_7 = 0
    write_instrument_read(result_7, 'result_7')
    print('malloc', sys.getsizeof(result_7), 'result_7')
    for i_7 in range(len(instrument_read(row_7, 'row_7'))):
        print(106, 91)
        diff_7 = instrument_read_sub(instrument_read(row_7, 'row_7'),
            'row_7', instrument_read(i_7, 'i_7'), None, None, False
            ) - get_mean(instrument_read(row_7, 'row_7'))
        write_instrument_read(diff_7, 'diff_7')
        print('malloc', sys.getsizeof(diff_7), 'diff_7')
        print(106, 92)
        result_7 += instrument_read(diff_7, 'diff_7') * instrument_read(diff_7,
            'diff_7')
        write_instrument_read(result_7, 'result_7')
    print('exit scope 7')
    return instrument_read(math, 'math').sqrt(instrument_read(result_7,
        'result_7') / len(instrument_read(row_7, 'row_7')))
    print('exit scope 7')


def BN_layer(img, channels, weights, biases):
    print('enter scope 8')
    print(1, 95)
    img_8 = instrument_read(img, 'img')
    write_instrument_read(img_8, 'img_8')
    print('malloc', sys.getsizeof(img_8), 'img_8')
    channels_8 = instrument_read(channels, 'channels')
    write_instrument_read(channels_8, 'channels_8')
    print('malloc', sys.getsizeof(channels_8), 'channels_8')
    weights_8 = instrument_read(weights, 'weights')
    write_instrument_read(weights_8, 'weights_8')
    print('malloc', sys.getsizeof(weights_8), 'weights_8')
    biases_8 = instrument_read(biases, 'biases')
    write_instrument_read(biases_8, 'biases_8')
    print('malloc', sys.getsizeof(biases_8), 'biases_8')
    for i_8 in range(instrument_read(channels_8, 'channels_8')):
        for j_8 in range(len(instrument_read_sub(instrument_read(img_8,
            'img_8'), 'img_8', 0, None, None, False))):
            print(114, 98)
            dev_8 = std_dev(instrument_read_sub(instrument_read_sub(
                instrument_read_sub(instrument_read(img_8, 'img_8'),
                'img_8', instrument_read(i_8, 'i_8'), None, None, False),
                'img_8[i_8]', instrument_read(j_8, 'j_8'), None, None,
                False), 'img_8[i_8][j_8]', None, None, None, True))
            write_instrument_read(dev_8, 'dev_8')
            print('malloc', sys.getsizeof(dev_8), 'dev_8')
            print(114, 99)
            mean_8 = get_mean(instrument_read_sub(instrument_read_sub(
                instrument_read_sub(instrument_read(img_8, 'img_8'),
                'img_8', instrument_read(i_8, 'i_8'), None, None, False),
                'img_8[i_8]', instrument_read(j_8, 'j_8'), None, None,
                False), 'img_8[i_8][j_8]', None, None, None, True))
            write_instrument_read(mean_8, 'mean_8')
            print('malloc', sys.getsizeof(mean_8), 'mean_8')
            if instrument_read(dev_8, 'dev_8') == 0.0:
                print(116, 100)
                dev_8 = 1.0
                write_instrument_read(dev_8, 'dev_8')
                print('malloc', sys.getsizeof(dev_8), 'dev_8')
            for k_8 in range(len(instrument_read_sub(instrument_read_sub(
                instrument_read(img_8, 'img_8'), 'img_8', 0, None, None,
                False), 'img_8[0]', 0, None, None, False))):
                print(118, 102)
                img_8[instrument_read(instrument_read(i_8, 'i_8'), 'i_8')][
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8')][
                    instrument_read(instrument_read(k_8, 'k_8'), 'k_8')
                    ] = instrument_read_sub(instrument_read(weights_8,
                    'weights_8'), 'weights_8', instrument_read(j_8, 'j_8'),
                    None, None, False) * ((instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (img_8, 'img_8'), 'img_8', instrument_read(i_8, 'i_8'),
                    None, None, False), 'img_8[i_8]', instrument_read(j_8,
                    'j_8'), None, None, False), 'img_8[i_8][j_8]',
                    instrument_read(k_8, 'k_8'), None, None, False) -
                    instrument_read(mean_8, 'mean_8')) / instrument_read(
                    dev_8, 'dev_8')) + instrument_read_sub(instrument_read(
                    biases_8, 'biases_8'), 'biases_8', instrument_read(j_8,
                    'j_8'), None, None, False)
                write_instrument_read_sub(img_8[instrument_read(
                    instrument_read(i_8, 'i_8'), 'i_8')][instrument_read(
                    instrument_read(j_8, 'j_8'), 'j_8')],
                    "img_8[instrument_read(i_8, 'i_8')][instrument_read(j_8, 'j_8')]"
                    , instrument_read(instrument_read(k_8, 'k_8'), 'k_8'),
                    None, None, False)
    print('exit scope 8')
    return instrument_read(img_8, 'img_8')
    print('exit scope 8')


def flatten_layer(img):
    print('enter scope 9')
    print(1, 105)
    img_9 = instrument_read(img, 'img')
    write_instrument_read(img_9, 'img_9')
    print('malloc', sys.getsizeof(img_9), 'img_9')
    print(123, 106)
    result_9 = instrument_read(np, 'np').zeros(len(instrument_read(img_9,
        'img_9')) * len(instrument_read_sub(instrument_read(img_9, 'img_9'),
        'img_9', 0, None, None, False)) * len(instrument_read_sub(
        instrument_read_sub(instrument_read(img_9, 'img_9'), 'img_9', 0,
        None, None, False), 'img_9[0]', 0, None, None, False)))
    write_instrument_read(result_9, 'result_9')
    print('malloc', sys.getsizeof(result_9), 'result_9')
    print(123, 107)
    index_9 = 0
    write_instrument_read(index_9, 'index_9')
    print('malloc', sys.getsizeof(index_9), 'index_9')
    for i_9 in range(len(instrument_read(img_9, 'img_9'))):
        for j_9 in range(len(instrument_read_sub(instrument_read(img_9,
            'img_9'), 'img_9', 0, None, None, False))):
            for k_9 in range(len(instrument_read_sub(instrument_read_sub(
                instrument_read(img_9, 'img_9'), 'img_9', 0, None, None,
                False), 'img_9[0]', 0, None, None, False))):
                print(129, 111)
                result_9[instrument_read(instrument_read(index_9, 'index_9'
                    ), 'index_9')] = instrument_read_sub(instrument_read_sub
                    (instrument_read_sub(instrument_read(img_9, 'img_9'),
                    'img_9', instrument_read(i_9, 'i_9'), None, None, False
                    ), 'img_9[i_9]', instrument_read(j_9, 'j_9'), None,
                    None, False), 'img_9[i_9][j_9]', instrument_read(k_9,
                    'k_9'), None, None, False)
                write_instrument_read_sub(result_9, 'result_9',
                    instrument_read(instrument_read(index_9, 'index_9'),
                    'index_9'), None, None, False)
                print(129, 112)
                index_9 += 1
                write_instrument_read(index_9, 'index_9')
    print('exit scope 9')
    return instrument_read(result_9, 'result_9')
    print('exit scope 9')


def fc_layer(arr, W, W_0):
    print('enter scope 10')
    print(1, 115)
    arr_10 = instrument_read(arr, 'arr')
    write_instrument_read(arr_10, 'arr_10')
    print('malloc', sys.getsizeof(arr_10), 'arr_10')
    W_10 = instrument_read(W, 'W')
    write_instrument_read(W_10, 'W_10')
    print('malloc', sys.getsizeof(W_10), 'W_10')
    W_0_10 = instrument_read(W_0, 'W_0')
    write_instrument_read(W_0_10, 'W_0_10')
    print('malloc', sys.getsizeof(W_0_10), 'W_0_10')
    print(134, 116)
    result_10 = instrument_read(np, 'np').zeros(len(instrument_read_sub(
        instrument_read(W_10, 'W_10'), 'W_10', 0, None, None, False)))
    write_instrument_read(result_10, 'result_10')
    print('malloc', sys.getsizeof(result_10), 'result_10')
    for i_10 in range(len(instrument_read_sub(instrument_read(W_10, 'W_10'),
        'W_10', 0, None, None, False))):
        print(136, 118)
        sum_val_10 = instrument_read_sub(instrument_read(W_0_10, 'W_0_10'),
            'W_0_10', instrument_read(i_10, 'i_10'), None, None, False)
        write_instrument_read(sum_val_10, 'sum_val_10')
        print('malloc', sys.getsizeof(sum_val_10), 'sum_val_10')
        for j_10 in range(len(instrument_read(arr_10, 'arr_10'))):
            print(139, 120)
            sum_val_10 += instrument_read_sub(instrument_read(arr_10,
                'arr_10'), 'arr_10', instrument_read(j_10, 'j_10'), None,
                None, False) * instrument_read_sub(instrument_read_sub(
                instrument_read(W_10, 'W_10'), 'W_10', instrument_read(j_10,
                'j_10'), None, None, False), 'W_10[j_10]', instrument_read(
                i_10, 'i_10'), None, None, False)
            write_instrument_read(sum_val_10, 'sum_val_10')
        print(140, 121)
        result_10[instrument_read(instrument_read(i_10, 'i_10'), 'i_10')
            ] = instrument_read(sum_val_10, 'sum_val_10')
        write_instrument_read_sub(result_10, 'result_10', instrument_read(
            instrument_read(i_10, 'i_10'), 'i_10'), None, None, False)
    print('exit scope 10')
    return instrument_read(result_10, 'result_10')
    print('exit scope 10')


def softmax(img):
    print('enter scope 11')
    print(1, 124)
    img_11 = instrument_read(img, 'img')
    write_instrument_read(img_11, 'img_11')
    print('malloc', sys.getsizeof(img_11), 'img_11')
    print(144, 125)
    sum_val_11 = 0
    write_instrument_read(sum_val_11, 'sum_val_11')
    print('malloc', sys.getsizeof(sum_val_11), 'sum_val_11')
    for i_11 in range(len(instrument_read(img_11, 'img_11'))):
        print(146, 127)
        sum_val_11 += instrument_read(math, 'math').exp(instrument_read_sub
            (instrument_read(img_11, 'img_11'), 'img_11', instrument_read(
            i_11, 'i_11'), None, None, False))
        write_instrument_read(sum_val_11, 'sum_val_11')
    print(147, 128)
    result_11 = instrument_read(np, 'np').zeros(len(instrument_read(img_11,
        'img_11')))
    write_instrument_read(result_11, 'result_11')
    print('malloc', sys.getsizeof(result_11), 'result_11')
    for i_11 in range(len(instrument_read(img_11, 'img_11'))):
        print(149, 130)
        result_11[instrument_read(instrument_read(i_11, 'i_11'), 'i_11')
            ] = instrument_read(math, 'math').exp(instrument_read_sub(
            instrument_read(img_11, 'img_11'), 'img_11', instrument_read(
            i_11, 'i_11'), None, None, False)) / instrument_read(sum_val_11,
            'sum_val_11')
        write_instrument_read_sub(result_11, 'result_11', instrument_read(
            instrument_read(i_11, 'i_11'), 'i_11'), None, None, False)
    print('exit scope 11')
    return instrument_read(result_11, 'result_11')
    print('exit scope 11')


def main():
    print('enter scope 12')
    print(1, 133)
    print(154, 134)
    zero_pad_12 = 3
    write_instrument_read(zero_pad_12, 'zero_pad_12')
    print('malloc', sys.getsizeof(zero_pad_12), 'zero_pad_12')
    print(154, 135)
    stride_12 = 2
    write_instrument_read(stride_12, 'stride_12')
    print('malloc', sys.getsizeof(stride_12), 'stride_12')
    print(154, 136)
    filt_12 = instrument_read(np, 'np').random.rand(3, 7, 7)
    write_instrument_read(filt_12, 'filt_12')
    print('malloc', sys.getsizeof(filt_12), 'filt_12')
    print(154, 137)
    img_12 = instrument_read(np, 'np').random.rand(3, 224, 224)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(154, 140)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, instrument_read(zero_pad_12, 'zero_pad_12'),
        instrument_read(stride_12, 'stride_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(154, 141)
    weights_12 = instrument_read(np, 'np').random.rand(len(
        instrument_read_sub(instrument_read(img_12, 'img_12'), 'img_12', 0,
        None, None, False)))
    write_instrument_read(weights_12, 'weights_12')
    print('malloc', sys.getsizeof(weights_12), 'weights_12')
    print(154, 142)
    biases_12 = instrument_read(np, 'np').random.rand(len(
        instrument_read_sub(instrument_read(img_12, 'img_12'), 'img_12', 0,
        None, None, False)))
    write_instrument_read(biases_12, 'biases_12')
    print('malloc', sys.getsizeof(biases_12), 'biases_12')
    print(154, 143)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(154, 144)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(154, 145)
    img_12 = max_pool(instrument_read(img_12, 'img_12'), 3, 3, 1, 2)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(154, 148)
    filt_12 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt_12, 'filt_12'), 'filt_12',
        0, None, None, False)), len(instrument_read_sub(instrument_read_sub
        (instrument_read(filt_12, 'filt_12'), 'filt_12', 0, None, None,
        False), 'filt_12[0]', 0, None, None, False)))
    write_instrument_read(filt_12, 'filt_12')
    print('malloc', sys.getsizeof(filt_12), 'filt_12')
    for i_12 in range(2):
        print(156, 150)
        byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
            'img_12'))
        write_instrument_read(byPass_12, 'byPass_12')
        print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
        print(156, 151)
        img_12 = conv_layer(instrument_read(img_12, 'img_12'),
            instrument_read(filt_12, 'filt_12'), 1, 3, 1)
        write_instrument_read(img_12, 'img_12')
        print('malloc', sys.getsizeof(img_12), 'img_12')
        print(156, 152)
        img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1,
            instrument_read(weights_12, 'weights_12'), instrument_read(
            biases_12, 'biases_12'))
        write_instrument_read(img_12, 'img_12')
        print('malloc', sys.getsizeof(img_12), 'img_12')
        print(156, 153)
        img_12 = reLU(instrument_read(img_12, 'img_12'))
        write_instrument_read(img_12, 'img_12')
        print('malloc', sys.getsizeof(img_12), 'img_12')
        print(156, 154)
        img_12 = conv_layer(instrument_read(img_12, 'img_12'),
            instrument_read(filt_12, 'filt_12'), 1, 3, 1)
        write_instrument_read(img_12, 'img_12')
        print('malloc', sys.getsizeof(img_12), 'img_12')
        print(156, 155)
        img_12 += instrument_read(byPass_12, 'byPass_12')
        write_instrument_read(img_12, 'img_12')
        print(156, 156)
        img_12 = reLU(instrument_read(img_12, 'img_12'))
        write_instrument_read(img_12, 'img_12')
        print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 159)
    filt_12 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt_12, 'filt_12'), 'filt_12',
        0, None, None, False)), len(instrument_read_sub(instrument_read_sub
        (instrument_read(filt_12, 'filt_12'), 'filt_12', 0, None, None,
        False), 'filt_12[0]', 0, None, None, False)))
    write_instrument_read(filt_12, 'filt_12')
    print('malloc', sys.getsizeof(filt_12), 'filt_12')
    print(157, 160)
    byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
        'img_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 161)
    byPass_12 = conv_layer(instrument_read(byPass_12, 'byPass_12'),
        instrument_read(filt_12, 'filt_12'), 1, 3, 2)
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 162)
    byPass_12 = BN_layer(instrument_read(byPass_12, 'byPass_12'), 1,
        instrument_read(weights_12, 'weights_12'), instrument_read(
        biases_12, 'biases_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 163)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 164)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 165)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 166)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 2)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 167)
    img_12 += instrument_read(byPass_12, 'byPass_12')
    write_instrument_read(img_12, 'img_12')
    print(157, 168)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 170)
    byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
        'img_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 171)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 172)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 173)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 174)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 175)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 176)
    img_12 += instrument_read(byPass_12, 'byPass_12')
    write_instrument_read(img_12, 'img_12')
    print(157, 177)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 180)
    filt_12 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt_12, 'filt_12'), 'filt_12',
        0, None, None, False)), len(instrument_read_sub(instrument_read_sub
        (instrument_read(filt_12, 'filt_12'), 'filt_12', 0, None, None,
        False), 'filt_12[0]', 0, None, None, False)))
    write_instrument_read(filt_12, 'filt_12')
    print('malloc', sys.getsizeof(filt_12), 'filt_12')
    print(157, 181)
    byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
        'img_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 182)
    byPass_12 = conv_layer(instrument_read(byPass_12, 'byPass_12'),
        instrument_read(filt_12, 'filt_12'), 1, 3, 2)
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 183)
    byPass_12 = BN_layer(instrument_read(byPass_12, 'byPass_12'), 1,
        instrument_read(weights_12, 'weights_12'), instrument_read(
        biases_12, 'biases_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 184)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 185)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 186)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 187)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 2)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 188)
    img_12 += instrument_read(byPass_12, 'byPass_12')
    write_instrument_read(img_12, 'img_12')
    print(157, 189)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 191)
    byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
        'img_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 192)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 193)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 194)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 195)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 196)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 197)
    img_12 += instrument_read(byPass_12, 'byPass_12')
    write_instrument_read(img_12, 'img_12')
    print(157, 198)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 201)
    filt_12 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt_12, 'filt_12'), 'filt_12',
        0, None, None, False)), len(instrument_read_sub(instrument_read_sub
        (instrument_read(filt_12, 'filt_12'), 'filt_12', 0, None, None,
        False), 'filt_12[0]', 0, None, None, False)))
    write_instrument_read(filt_12, 'filt_12')
    print('malloc', sys.getsizeof(filt_12), 'filt_12')
    print(157, 202)
    byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
        'img_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 203)
    byPass_12 = conv_layer(instrument_read(byPass_12, 'byPass_12'),
        instrument_read(filt_12, 'filt_12'), 1, 3, 2)
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 204)
    byPass_12 = BN_layer(instrument_read(byPass_12, 'byPass_12'), 1,
        instrument_read(weights_12, 'weights_12'), instrument_read(
        biases_12, 'biases_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 205)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 206)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 207)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 208)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 2)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 209)
    img_12 += instrument_read(byPass_12, 'byPass_12')
    write_instrument_read(img_12, 'img_12')
    print(157, 210)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 212)
    byPass_12 = instrument_read(np, 'np').copy(instrument_read(img_12,
        'img_12'))
    write_instrument_read(byPass_12, 'byPass_12')
    print('malloc', sys.getsizeof(byPass_12), 'byPass_12')
    print(157, 213)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 214)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 215)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 216)
    img_12 = conv_layer(instrument_read(img_12, 'img_12'), instrument_read(
        filt_12, 'filt_12'), 1, 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 217)
    img_12 = BN_layer(instrument_read(img_12, 'img_12'), 1, instrument_read
        (weights_12, 'weights_12'), instrument_read(biases_12, 'biases_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 218)
    img_12 += instrument_read(byPass_12, 'byPass_12')
    write_instrument_read(img_12, 'img_12')
    print(157, 219)
    img_12 = reLU(instrument_read(img_12, 'img_12'))
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 220)
    img_12 = avg_pool(instrument_read(img_12, 'img_12'), len(
        instrument_read_sub(instrument_read(img_12, 'img_12'), 'img_12', 0,
        None, None, False)), len(instrument_read_sub(instrument_read_sub(
        instrument_read(img_12, 'img_12'), 'img_12', 0, None, None, False),
        'img_12[0]', 0, None, None, False)), 3, 1)
    write_instrument_read(img_12, 'img_12')
    print('malloc', sys.getsizeof(img_12), 'img_12')
    print(157, 221)
    flat_12 = flatten_layer(instrument_read(img_12, 'img_12'))
    write_instrument_read(flat_12, 'flat_12')
    print('malloc', sys.getsizeof(flat_12), 'flat_12')
    print(157, 223)
    weights_12 = instrument_read(np, 'np').random.rand(len(instrument_read(
        img_12, 'img_12')) * len(instrument_read_sub(instrument_read(img_12,
        'img_12'), 'img_12', 0, None, None, False)) * len(
        instrument_read_sub(instrument_read_sub(instrument_read(img_12,
        'img_12'), 'img_12', 0, None, None, False), 'img_12[0]', 0, None,
        None, False)), 7)
    write_instrument_read(weights_12, 'weights_12')
    print('malloc', sys.getsizeof(weights_12), 'weights_12')
    print(157, 224)
    w_0_12 = instrument_read(np, 'np').random.rand(7)
    write_instrument_read(w_0_12, 'w_0_12')
    print('malloc', sys.getsizeof(w_0_12), 'w_0_12')
    print(157, 225)
    flat_12 = fc_layer(instrument_read(flat_12, 'flat_12'), instrument_read
        (weights_12, 'weights_12'), instrument_read(w_0_12, 'w_0_12'))
    write_instrument_read(flat_12, 'flat_12')
    print('malloc', sys.getsizeof(flat_12), 'flat_12')
    print(157, 226)
    final_12 = softmax(instrument_read(flat_12, 'flat_12'))
    write_instrument_read(final_12, 'final_12')
    print('malloc', sys.getsizeof(final_12), 'final_12')
    print('exit scope 12')
    return 0
    print('exit scope 12')


if instrument_read(__name__, '__name__') == '__main__':
    instrument_read(loop, 'loop').start_unroll
    main()
