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
    img__1 = instrument_read(img, 'img')
    write_instrument_read(img__1, 'img__1')
    print('malloc', sys.getsizeof(img__1), 'img__1')
    zero_pad__1 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad__1, 'zero_pad__1')
    print('malloc', sys.getsizeof(zero_pad__1), 'zero_pad__1')
    print(3, 6)
    len_new__1 = len(instrument_read_sub(instrument_read(img__1, 'img__1'),
        'img__1', 0, None, None, False)) + 2 * instrument_read(zero_pad__1,
        'zero_pad__1')
    write_instrument_read(len_new__1, 'len_new__1')
    print('malloc', sys.getsizeof(len_new__1), 'len_new__1')
    print(3, 7)
    wid_new__1 = len(instrument_read_sub(instrument_read_sub(
        instrument_read(img__1, 'img__1'), 'img__1', 0, None, None, False),
        'img__1[0]', 0, None, None, False)) + 2 * instrument_read(zero_pad__1,
        'zero_pad__1')
    write_instrument_read(wid_new__1, 'wid_new__1')
    print('malloc', sys.getsizeof(wid_new__1), 'wid_new__1')
    print(3, 8)
    new_img__1 = instrument_read(np, 'np').zeros((len(instrument_read(
        img__1, 'img__1')), instrument_read(len_new__1, 'len_new__1'),
        instrument_read(wid_new__1, 'wid_new__1')))
    write_instrument_read(new_img__1, 'new_img__1')
    print('malloc', sys.getsizeof(new_img__1), 'new_img__1')
    for i__1 in range(len(instrument_read(img__1, 'img__1'))):
        print('enter scope 2')
        for j__2 in range(instrument_read(len_new__1, 'len_new__1')):
            print('enter scope 3')
            print(7, 11)
            make_zero__3 = instrument_read(j__2, 'j__2') < instrument_read(
                zero_pad__1, 'zero_pad__1') or instrument_read(j__2, 'j__2'
                ) >= instrument_read(len_new__1, 'len_new__1'
                ) - instrument_read(zero_pad__1, 'zero_pad__1')
            write_instrument_read(make_zero__3, 'make_zero__3')
            print('malloc', sys.getsizeof(make_zero__3), 'make_zero__3')
            for k__3 in range(instrument_read(wid_new__1, 'wid_new__1')):
                print('enter scope 4')
                if instrument_read(k__3, 'k__3') < instrument_read(zero_pad__1,
                    'zero_pad__1') or instrument_read(k__3, 'k__3'
                    ) >= instrument_read(wid_new__1, 'wid_new__1'
                    ) - instrument_read(zero_pad__1, 'zero_pad__1'
                    ) or instrument_read(make_zero__3, 'make_zero__3'):
                    print(12, 14)
                    new_img__1[instrument_read(instrument_read(i__1, 'i__1'
                        ), 'i__1')][instrument_read(instrument_read(j__2,
                        'j__2'), 'j__2')][instrument_read(instrument_read(
                        k__3, 'k__3'), 'k__3')] = 0
                    write_instrument_read_sub(new_img__1[instrument_read(
                        instrument_read(i__1, 'i__1'), 'i__1')][
                        instrument_read(instrument_read(j__2, 'j__2'),
                        'j__2')],
                        "new_img__1[instrument_read(i__1, 'i__1')][instrument_read(j__2, 'j__2')]"
                        , instrument_read(instrument_read(k__3, 'k__3'),
                        'k__3'), None, None, False)
                else:
                    print(14, 16)
                    new_img__1[instrument_read(instrument_read(i__1, 'i__1'
                        ), 'i__1')][instrument_read(instrument_read(j__2,
                        'j__2'), 'j__2')][instrument_read(instrument_read(
                        k__3, 'k__3'), 'k__3')] = instrument_read_sub(
                        instrument_read_sub(instrument_read_sub(
                        instrument_read(img__1, 'img__1'), 'img__1',
                        instrument_read(i__1, 'i__1'), None, None, False),
                        'img__1[i__1]', instrument_read(j__2, 'j__2') -
                        instrument_read(zero_pad__1, 'zero_pad__1'), None,
                        None, False), 'img__1[i__1][j__2 - zero_pad__1]', 
                        instrument_read(k__3, 'k__3') - instrument_read(
                        zero_pad__1, 'zero_pad__1'), None, None, False)
                    write_instrument_read_sub(new_img__1[instrument_read(
                        instrument_read(i__1, 'i__1'), 'i__1')][
                        instrument_read(instrument_read(j__2, 'j__2'),
                        'j__2')],
                        "new_img__1[instrument_read(i__1, 'i__1')][instrument_read(j__2, 'j__2')]"
                        , instrument_read(instrument_read(k__3, 'k__3'),
                        'k__3'), None, None, False)
                print('exit scope 4')
            print('exit scope 3')
        print('exit scope 2')
    print('exit scope 1')
    return instrument_read(new_img__1, 'new_img__1')
    print('exit scope 1')


def conv_layer(img, filt, numFilt, zero_pad, stride):
    print('enter scope 5')
    print(1, 20)
    img__5 = instrument_read(img, 'img')
    write_instrument_read(img__5, 'img__5')
    print('malloc', sys.getsizeof(img__5), 'img__5')
    filt__5 = instrument_read(filt, 'filt')
    write_instrument_read(filt__5, 'filt__5')
    print('malloc', sys.getsizeof(filt__5), 'filt__5')
    numFilt__5 = instrument_read(numFilt, 'numFilt')
    write_instrument_read(numFilt__5, 'numFilt__5')
    print('malloc', sys.getsizeof(numFilt__5), 'numFilt__5')
    zero_pad__5 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad__5, 'zero_pad__5')
    print('malloc', sys.getsizeof(zero_pad__5), 'zero_pad__5')
    stride__5 = instrument_read(stride, 'stride')
    write_instrument_read(stride__5, 'stride__5')
    print('malloc', sys.getsizeof(stride__5), 'stride__5')
    print(18, 21)
    f_len__5 = int((len(instrument_read_sub(instrument_read(img__5,
        'img__5'), 'img__5', 0, None, None, False)) - len(
        instrument_read_sub(instrument_read(filt__5, 'filt__5'), 'filt__5',
        0, None, None, False)) + 2 * instrument_read(zero_pad__5,
        'zero_pad__5')) / instrument_read(stride__5, 'stride__5') + 1)
    write_instrument_read(f_len__5, 'f_len__5')
    print('malloc', sys.getsizeof(f_len__5), 'f_len__5')
    print(18, 22)
    f_wid__5 = int((len(instrument_read_sub(instrument_read_sub(
        instrument_read(img__5, 'img__5'), 'img__5', 0, None, None, False),
        'img__5[0]', 0, None, None, False)) - len(instrument_read_sub(
        instrument_read_sub(instrument_read(filt__5, 'filt__5'), 'filt__5',
        0, None, None, False), 'filt__5[0]', 0, None, None, False)) + 2 *
        instrument_read(zero_pad__5, 'zero_pad__5')) / instrument_read(
        stride__5, 'stride__5') + 1)
    write_instrument_read(f_wid__5, 'f_wid__5')
    print('malloc', sys.getsizeof(f_wid__5), 'f_wid__5')
    print(18, 23)
    biases__5 = instrument_read(np, 'np').random.rand(instrument_read(
        f_len__5, 'f_len__5'))
    write_instrument_read(biases__5, 'biases__5')
    print('malloc', sys.getsizeof(biases__5), 'biases__5')
    print(18, 24)
    img_new__5 = zero_pad_arr(instrument_read(img__5, 'img__5'),
        instrument_read(zero_pad__5, 'zero_pad__5'))
    write_instrument_read(img_new__5, 'img_new__5')
    print('malloc', sys.getsizeof(img_new__5), 'img_new__5')
    print(18, 25)
    f_new__5 = instrument_read(np, 'np').zeros((instrument_read(numFilt__5,
        'numFilt__5'), instrument_read(f_len__5, 'f_len__5'),
        instrument_read(f_wid__5, 'f_wid__5')))
    write_instrument_read(f_new__5, 'f_new__5')
    print('malloc', sys.getsizeof(f_new__5), 'f_new__5')
    for i__5 in range(instrument_read(numFilt__5, 'numFilt__5')):
        print('enter scope 6')
        for j__6 in range(instrument_read(f_len__5, 'f_len__5')):
            print('enter scope 7')
            for k__7 in range(instrument_read(f_wid__5, 'f_wid__5')):
                print('enter scope 8')
                for l__8 in range(len(instrument_read(filt__5, 'filt__5'))):
                    print('enter scope 9')
                    for c__9 in range(len(instrument_read_sub(
                        instrument_read(filt__5, 'filt__5'), 'filt__5', 0,
                        None, None, False))):
                        print('enter scope 10')
                        for d__10 in range(len(instrument_read_sub(
                            instrument_read_sub(instrument_read(filt__5,
                            'filt__5'), 'filt__5', 0, None, None, False),
                            'filt__5[0]', 0, None, None, False))):
                            print('enter scope 11')
                            print(30, 32)
                            f_new__5[instrument_read(i__5, 'i__5')][
                                instrument_read(j__6, 'j__6')][instrument_read
                                (k__7, 'k__7')] += instrument_read_sub(
                                instrument_read_sub(instrument_read_sub(
                                instrument_read(img_new__5, 'img_new__5'),
                                'img_new__5', instrument_read(l__8, 'l__8'),
                                None, None, False), 'img_new__5[l__8]', 
                                instrument_read(j__6, 'j__6') *
                                instrument_read(stride__5, 'stride__5') +
                                instrument_read(c__9, 'c__9'), None, None,
                                False),
                                'img_new__5[l__8][j__6 * stride__5 + c__9]',
                                instrument_read(k__7, 'k__7') *
                                instrument_read(stride__5, 'stride__5') +
                                instrument_read(d__10, 'd__10'), None, None,
                                False) * instrument_read_sub(
                                instrument_read_sub(instrument_read_sub(
                                instrument_read(filt__5, 'filt__5'),
                                'filt__5', instrument_read(l__8, 'l__8'),
                                None, None, False), 'filt__5[l__8]',
                                instrument_read(c__9, 'c__9'), None, None,
                                False), 'filt__5[l__8][c__9]',
                                instrument_read(d__10, 'd__10'), None, None,
                                False)
                            write_instrument_read_sub(f_new__5[
                                instrument_read(instrument_read(i__5,
                                'i__5'), 'i__5')][instrument_read(
                                instrument_read(j__6, 'j__6'), 'j__6')],
                                "f_new__5[instrument_read(i__5, 'i__5')][instrument_read(j__6, 'j__6')]"
                                , instrument_read(instrument_read(k__7,
                                'k__7'), 'k__7'), None, None, False)
                            print('exit scope 11')
                        print('exit scope 10')
                    print('exit scope 9')
                print('exit scope 8')
            print('exit scope 7')
        print('exit scope 6')
    for i__5 in range(instrument_read(numFilt__5, 'numFilt__5')):
        print('enter scope 12')
        for j__12 in range(instrument_read(f_len__5, 'f_len__5')):
            print('enter scope 13')
            for k__13 in range(instrument_read(f_wid__5, 'f_wid__5')):
                print('enter scope 14')
                print(36, 36)
                f_new__5[instrument_read(i__5, 'i__5')][instrument_read(
                    j__12, 'j__12')][instrument_read(k__13, 'k__13')
                    ] += instrument_read_sub(instrument_read(biases__5,
                    'biases__5'), 'biases__5', instrument_read(j__12,
                    'j__12'), None, None, False)
                write_instrument_read_sub(f_new__5[instrument_read(
                    instrument_read(i__5, 'i__5'), 'i__5')][instrument_read
                    (instrument_read(j__12, 'j__12'), 'j__12')],
                    "f_new__5[instrument_read(i__5, 'i__5')][instrument_read(j__12, 'j__12')]"
                    , instrument_read(instrument_read(k__13, 'k__13'),
                    'k__13'), None, None, False)
                print('exit scope 14')
            print('exit scope 13')
        print('exit scope 12')
    print('exit scope 5')
    return instrument_read(f_new__5, 'f_new__5')
    print('exit scope 5')


def max_pool(input, l, w, zero_pad, stride):
    print('enter scope 15')
    print(1, 39)
    input__15 = instrument_read(input, 'input')
    write_instrument_read(input__15, 'input__15')
    print('malloc', sys.getsizeof(input__15), 'input__15')
    l__15 = instrument_read(l, 'l')
    write_instrument_read(l__15, 'l__15')
    print('malloc', sys.getsizeof(l__15), 'l__15')
    w__15 = instrument_read(w, 'w')
    write_instrument_read(w__15, 'w__15')
    print('malloc', sys.getsizeof(w__15), 'w__15')
    zero_pad__15 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad__15, 'zero_pad__15')
    print('malloc', sys.getsizeof(zero_pad__15), 'zero_pad__15')
    stride__15 = instrument_read(stride, 'stride')
    write_instrument_read(stride__15, 'stride__15')
    print('malloc', sys.getsizeof(stride__15), 'stride__15')
    if instrument_read(zero_pad__15, 'zero_pad__15') > 0:
        print(42, 41)
        input__15 = zero_pad_arr(instrument_read(input__15, 'input__15'),
            instrument_read(zero_pad__15, 'zero_pad__15'))
        write_instrument_read(input__15, 'input__15')
        print('malloc', sys.getsizeof(input__15), 'input__15')
    print(43, 42)
    res_l__15 = int((len(instrument_read_sub(instrument_read(input__15,
        'input__15'), 'input__15', 0, None, None, False)) - instrument_read
        (l__15, 'l__15')) / instrument_read(stride__15, 'stride__15') + 1)
    write_instrument_read(res_l__15, 'res_l__15')
    print('malloc', sys.getsizeof(res_l__15), 'res_l__15')
    print(43, 43)
    res_w__15 = int((len(instrument_read_sub(instrument_read_sub(
        instrument_read(input__15, 'input__15'), 'input__15', 0, None, None,
        False), 'input__15[0]', 0, None, None, False)) - instrument_read(
        w__15, 'w__15')) / instrument_read(stride__15, 'stride__15') + 1)
    write_instrument_read(res_w__15, 'res_w__15')
    print('malloc', sys.getsizeof(res_w__15), 'res_w__15')
    print(43, 44)
    result__15 = instrument_read(np, 'np').zeros((len(instrument_read(
        input__15, 'input__15')), instrument_read(res_l__15, 'res_l__15'),
        instrument_read(res_w__15, 'res_w__15')))
    write_instrument_read(result__15, 'result__15')
    print('malloc', sys.getsizeof(result__15), 'result__15')
    for i__15 in range(len(instrument_read(input__15, 'input__15'))):
        print('enter scope 16')
        for j__16 in range(instrument_read(res_l__15, 'res_l__15')):
            print('enter scope 17')
            for k__17 in range(instrument_read(res_w__15, 'res_w__15')):
                print('enter scope 18')
                for c__18 in range(instrument_read(l__15, 'l__15')):
                    print('enter scope 19')
                    for d__19 in range(instrument_read(w__15, 'w__15')):
                        print('enter scope 20')
                        print(53, 50)
                        result__15[instrument_read(instrument_read(i__15,
                            'i__15'), 'i__15')][instrument_read(
                            instrument_read(j__16, 'j__16'), 'j__16')][
                            instrument_read(instrument_read(k__17, 'k__17'),
                            'k__17')] = max(instrument_read_sub(
                            instrument_read_sub(instrument_read_sub(
                            instrument_read(input__15, 'input__15'),
                            'input__15', instrument_read(i__15, 'i__15'),
                            None, None, False), 'input__15[i__15]',
                            instrument_read(j__16, 'j__16'), None, None,
                            False), 'input__15[i__15][j__16]',
                            instrument_read(k__17, 'k__17'), None, None,
                            False), instrument_read_sub(instrument_read_sub
                            (instrument_read_sub(instrument_read(result__15,
                            'result__15'), 'result__15', instrument_read(
                            i__15, 'i__15'), None, None, False),
                            'result__15[i__15]', instrument_read(j__16,
                            'j__16'), None, None, False),
                            'result__15[i__15][j__16]', instrument_read(
                            k__17, 'k__17'), None, None, False),
                            instrument_read_sub(instrument_read_sub(
                            instrument_read_sub(instrument_read(input__15,
                            'input__15'), 'input__15', instrument_read(
                            i__15, 'i__15'), None, None, False),
                            'input__15[i__15]', instrument_read(j__16,
                            'j__16') * instrument_read(stride__15,
                            'stride__15') + instrument_read(c__18, 'c__18'),
                            None, None, False),
                            'input__15[i__15][j__16 * stride__15 + c__18]',
                            instrument_read(k__17, 'k__17') *
                            instrument_read(stride__15, 'stride__15') +
                            instrument_read(d__19, 'd__19'), None, None, False)
                            )
                        write_instrument_read_sub(result__15[
                            instrument_read(instrument_read(i__15, 'i__15'),
                            'i__15')][instrument_read(instrument_read(j__16,
                            'j__16'), 'j__16')],
                            "result__15[instrument_read(i__15, 'i__15')][instrument_read(j__16, 'j__16')]"
                            , instrument_read(instrument_read(k__17,
                            'k__17'), 'k__17'), None, None, False)
                        print('exit scope 20')
                    print('exit scope 19')
                print('exit scope 18')
            print('exit scope 17')
        print('exit scope 16')
    for i__15 in range(len(instrument_read(input__15, 'input__15'))):
        print('enter scope 21')
        for j__21 in range(instrument_read(res_l__15, 'res_l__15')):
            print('enter scope 22')
            for k__22 in range(instrument_read(res_w__15, 'res_w__15')):
                print('enter scope 23')
                print(59, 54)
                result__15[instrument_read(i__15, 'i__15')][instrument_read
                    (j__21, 'j__21')][instrument_read(k__22, 'k__22')
                    ] /= instrument_read(l__15, 'l__15') * instrument_read(
                    w__15, 'w__15')
                write_instrument_read_sub(result__15[instrument_read(
                    instrument_read(i__15, 'i__15'), 'i__15')][
                    instrument_read(instrument_read(j__21, 'j__21'),
                    'j__21')],
                    "result__15[instrument_read(i__15, 'i__15')][instrument_read(j__21, 'j__21')]"
                    , instrument_read(instrument_read(k__22, 'k__22'),
                    'k__22'), None, None, False)
                print('exit scope 23')
            print('exit scope 22')
        print('exit scope 21')
    print('exit scope 15')
    return instrument_read(result__15, 'result__15')
    print('exit scope 15')


def avg_pool(input, l, w, zero_pad, stride):
    print('enter scope 24')
    print(1, 57)
    input__24 = instrument_read(input, 'input')
    write_instrument_read(input__24, 'input__24')
    print('malloc', sys.getsizeof(input__24), 'input__24')
    l__24 = instrument_read(l, 'l')
    write_instrument_read(l__24, 'l__24')
    print('malloc', sys.getsizeof(l__24), 'l__24')
    w__24 = instrument_read(w, 'w')
    write_instrument_read(w__24, 'w__24')
    print('malloc', sys.getsizeof(w__24), 'w__24')
    zero_pad__24 = instrument_read(zero_pad, 'zero_pad')
    write_instrument_read(zero_pad__24, 'zero_pad__24')
    print('malloc', sys.getsizeof(zero_pad__24), 'zero_pad__24')
    stride__24 = instrument_read(stride, 'stride')
    write_instrument_read(stride__24, 'stride__24')
    print('malloc', sys.getsizeof(stride__24), 'stride__24')
    if instrument_read(zero_pad__24, 'zero_pad__24') > 0:
        print(65, 59)
        input__24 = zero_pad_arr(instrument_read(input__24, 'input__24'),
            instrument_read(zero_pad__24, 'zero_pad__24'))
        write_instrument_read(input__24, 'input__24')
        print('malloc', sys.getsizeof(input__24), 'input__24')
    print(66, 60)
    res_l__24 = int((len(instrument_read_sub(instrument_read(input__24,
        'input__24'), 'input__24', 0, None, None, False)) - instrument_read
        (l__24, 'l__24')) / instrument_read(stride__24, 'stride__24') + 1)
    write_instrument_read(res_l__24, 'res_l__24')
    print('malloc', sys.getsizeof(res_l__24), 'res_l__24')
    print(66, 61)
    res_w__24 = int((len(instrument_read_sub(instrument_read_sub(
        instrument_read(input__24, 'input__24'), 'input__24', 0, None, None,
        False), 'input__24[0]', 0, None, None, False)) - instrument_read(
        w__24, 'w__24')) / instrument_read(stride__24, 'stride__24') + 1)
    write_instrument_read(res_w__24, 'res_w__24')
    print('malloc', sys.getsizeof(res_w__24), 'res_w__24')
    print(66, 62)
    result__24 = instrument_read(np, 'np').zeros((len(instrument_read(
        input__24, 'input__24')), instrument_read(res_l__24, 'res_l__24'),
        instrument_read(res_w__24, 'res_w__24')))
    write_instrument_read(result__24, 'result__24')
    print('malloc', sys.getsizeof(result__24), 'result__24')
    for i__24 in range(len(instrument_read(input__24, 'input__24'))):
        print('enter scope 25')
        for j__25 in range(instrument_read(res_l__24, 'res_l__24')):
            print('enter scope 26')
            for k__26 in range(instrument_read(res_w__24, 'res_w__24')):
                print('enter scope 27')
                for c__27 in range(instrument_read(l__24, 'l__24')):
                    print('enter scope 28')
                    for d__28 in range(instrument_read(w__24, 'w__24')):
                        print('enter scope 29')
                        print(76, 68)
                        result__24[instrument_read(i__24, 'i__24')][
                            instrument_read(j__25, 'j__25')][instrument_read
                            (k__26, 'k__26')] += instrument_read_sub(
                            instrument_read_sub(instrument_read_sub(
                            instrument_read(input__24, 'input__24'),
                            'input__24', instrument_read(i__24, 'i__24'),
                            None, None, False), 'input__24[i__24]', 
                            instrument_read(j__25, 'j__25') *
                            instrument_read(stride__24, 'stride__24') +
                            instrument_read(c__27, 'c__27'), None, None,
                            False),
                            'input__24[i__24][j__25 * stride__24 + c__27]',
                            instrument_read(k__26, 'k__26') *
                            instrument_read(stride__24, 'stride__24') +
                            instrument_read(d__28, 'd__28'), None, None, False)
                        write_instrument_read_sub(result__24[
                            instrument_read(instrument_read(i__24, 'i__24'),
                            'i__24')][instrument_read(instrument_read(j__25,
                            'j__25'), 'j__25')],
                            "result__24[instrument_read(i__24, 'i__24')][instrument_read(j__25, 'j__25')]"
                            , instrument_read(instrument_read(k__26,
                            'k__26'), 'k__26'), None, None, False)
                        print('exit scope 29')
                    print('exit scope 28')
                print('exit scope 27')
            print('exit scope 26')
        print('exit scope 25')
    for i__24 in range(len(instrument_read(input__24, 'input__24'))):
        print('enter scope 30')
        for j__30 in range(instrument_read(res_l__24, 'res_l__24')):
            print('enter scope 31')
            for k__31 in range(instrument_read(res_w__24, 'res_w__24')):
                print('enter scope 32')
                print(82, 72)
                result__24[instrument_read(i__24, 'i__24')][instrument_read
                    (j__30, 'j__30')][instrument_read(k__31, 'k__31')
                    ] /= instrument_read(l__24, 'l__24') * instrument_read(
                    w__24, 'w__24')
                write_instrument_read_sub(result__24[instrument_read(
                    instrument_read(i__24, 'i__24'), 'i__24')][
                    instrument_read(instrument_read(j__30, 'j__30'),
                    'j__30')],
                    "result__24[instrument_read(i__24, 'i__24')][instrument_read(j__30, 'j__30')]"
                    , instrument_read(instrument_read(k__31, 'k__31'),
                    'k__31'), None, None, False)
                print('exit scope 32')
            print('exit scope 31')
        print('exit scope 30')
    print('exit scope 24')
    return instrument_read(result__24, 'result__24')
    print('exit scope 24')


def reLU(img):
    print('enter scope 33')
    print(1, 75)
    img__33 = instrument_read(img, 'img')
    write_instrument_read(img__33, 'img__33')
    print('malloc', sys.getsizeof(img__33), 'img__33')
    for i__33 in range(len(instrument_read(img__33, 'img__33'))):
        print('enter scope 34')
        for j__34 in range(len(instrument_read_sub(instrument_read(img__33,
            'img__33'), 'img__33', 0, None, None, False))):
            print('enter scope 35')
            for k__35 in range(len(instrument_read_sub(instrument_read_sub(
                instrument_read(img__33, 'img__33'), 'img__33', 0, None,
                None, False), 'img__33[0]', 0, None, None, False))):
                print('enter scope 36')
                print(92, 79)
                img__33[instrument_read(instrument_read(i__33, 'i__33'),
                    'i__33')][instrument_read(instrument_read(j__34,
                    'j__34'), 'j__34')][instrument_read(instrument_read(
                    k__35, 'k__35'), 'k__35')] = max(instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (img__33, 'img__33'), 'img__33', instrument_read(i__33,
                    'i__33'), None, None, False), 'img__33[i__33]',
                    instrument_read(j__34, 'j__34'), None, None, False),
                    'img__33[i__33][j__34]', instrument_read(k__35, 'k__35'
                    ), None, None, False), 0)
                write_instrument_read_sub(img__33[instrument_read(
                    instrument_read(i__33, 'i__33'), 'i__33')][
                    instrument_read(instrument_read(j__34, 'j__34'),
                    'j__34')],
                    "img__33[instrument_read(i__33, 'i__33')][instrument_read(j__34, 'j__34')]"
                    , instrument_read(instrument_read(k__35, 'k__35'),
                    'k__35'), None, None, False)
                print('exit scope 36')
            print('exit scope 35')
        print('exit scope 34')
    print('exit scope 33')
    return instrument_read(img__33, 'img__33')
    print('exit scope 33')


def get_mean(row):
    print('enter scope 37')
    print(1, 82)
    row__37 = instrument_read(row, 'row')
    write_instrument_read(row__37, 'row__37')
    print('malloc', sys.getsizeof(row__37), 'row__37')
    print(97, 83)
    sum_val__37 = 0
    write_instrument_read(sum_val__37, 'sum_val__37')
    print('malloc', sys.getsizeof(sum_val__37), 'sum_val__37')
    for i__37 in range(len(instrument_read(row__37, 'row__37'))):
        print('enter scope 38')
        print(99, 85)
        sum_val__37 += instrument_read_sub(instrument_read(row__37,
            'row__37'), 'row__37', instrument_read(i__37, 'i__37'), None,
            None, False)
        write_instrument_read(sum_val__37, 'sum_val__37')
        print('exit scope 38')
    print('exit scope 37')
    return instrument_read(sum_val__37, 'sum_val__37') / len(instrument_read
        (row__37, 'row__37'))
    print('exit scope 37')


def std_dev(row):
    print('enter scope 39')
    print(1, 88)
    row__39 = instrument_read(row, 'row')
    write_instrument_read(row__39, 'row__39')
    print('malloc', sys.getsizeof(row__39), 'row__39')
    print(104, 89)
    result__39 = 0
    write_instrument_read(result__39, 'result__39')
    print('malloc', sys.getsizeof(result__39), 'result__39')
    for i__39 in range(len(instrument_read(row__39, 'row__39'))):
        print('enter scope 40')
        print(106, 91)
        diff__40 = instrument_read_sub(instrument_read(row__39, 'row__39'),
            'row__39', instrument_read(i__39, 'i__39'), None, None, False
            ) - get_mean(instrument_read(row__39, 'row__39'))
        write_instrument_read(diff__40, 'diff__40')
        print('malloc', sys.getsizeof(diff__40), 'diff__40')
        print(106, 92)
        result__39 += instrument_read(diff__40, 'diff__40') * instrument_read(
            diff__40, 'diff__40')
        write_instrument_read(result__39, 'result__39')
        print('exit scope 40')
    print('exit scope 39')
    return instrument_read(math, 'math').sqrt(instrument_read(result__39,
        'result__39') / len(instrument_read(row__39, 'row__39')))
    print('exit scope 39')


def BN_layer(img, channels, weights, biases):
    print('enter scope 41')
    print(1, 95)
    img__41 = instrument_read(img, 'img')
    write_instrument_read(img__41, 'img__41')
    print('malloc', sys.getsizeof(img__41), 'img__41')
    channels__41 = instrument_read(channels, 'channels')
    write_instrument_read(channels__41, 'channels__41')
    print('malloc', sys.getsizeof(channels__41), 'channels__41')
    weights__41 = instrument_read(weights, 'weights')
    write_instrument_read(weights__41, 'weights__41')
    print('malloc', sys.getsizeof(weights__41), 'weights__41')
    biases__41 = instrument_read(biases, 'biases')
    write_instrument_read(biases__41, 'biases__41')
    print('malloc', sys.getsizeof(biases__41), 'biases__41')
    for i__41 in range(instrument_read(channels__41, 'channels__41')):
        print('enter scope 42')
        for j__42 in range(len(instrument_read_sub(instrument_read(img__41,
            'img__41'), 'img__41', 0, None, None, False))):
            print('enter scope 43')
            print(114, 98)
            dev__43 = std_dev(instrument_read_sub(instrument_read_sub(
                instrument_read_sub(instrument_read(img__41, 'img__41'),
                'img__41', instrument_read(i__41, 'i__41'), None, None,
                False), 'img__41[i__41]', instrument_read(j__42, 'j__42'),
                None, None, False), 'img__41[i__41][j__42]', None, None,
                None, True))
            write_instrument_read(dev__43, 'dev__43')
            print('malloc', sys.getsizeof(dev__43), 'dev__43')
            print(114, 99)
            mean__43 = get_mean(instrument_read_sub(instrument_read_sub(
                instrument_read_sub(instrument_read(img__41, 'img__41'),
                'img__41', instrument_read(i__41, 'i__41'), None, None,
                False), 'img__41[i__41]', instrument_read(j__42, 'j__42'),
                None, None, False), 'img__41[i__41][j__42]', None, None,
                None, True))
            write_instrument_read(mean__43, 'mean__43')
            print('malloc', sys.getsizeof(mean__43), 'mean__43')
            if instrument_read(dev__43, 'dev__43') == 0.0:
                print(116, 100)
                dev__43 = 1.0
                write_instrument_read(dev__43, 'dev__43')
                print('malloc', sys.getsizeof(dev__43), 'dev__43')
            for k__43 in range(len(instrument_read_sub(instrument_read_sub(
                instrument_read(img__41, 'img__41'), 'img__41', 0, None,
                None, False), 'img__41[0]', 0, None, None, False))):
                print('enter scope 44')
                print(118, 102)
                img__41[instrument_read(instrument_read(i__41, 'i__41'),
                    'i__41')][instrument_read(instrument_read(j__42,
                    'j__42'), 'j__42')][instrument_read(instrument_read(
                    k__43, 'k__43'), 'k__43')] = instrument_read_sub(
                    instrument_read(weights__41, 'weights__41'),
                    'weights__41', instrument_read(j__42, 'j__42'), None,
                    None, False) * ((instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (img__41, 'img__41'), 'img__41', instrument_read(i__41,
                    'i__41'), None, None, False), 'img__41[i__41]',
                    instrument_read(j__42, 'j__42'), None, None, False),
                    'img__41[i__41][j__42]', instrument_read(k__43, 'k__43'
                    ), None, None, False) - instrument_read(mean__43,
                    'mean__43')) / instrument_read(dev__43, 'dev__43')
                    ) + instrument_read_sub(instrument_read(biases__41,
                    'biases__41'), 'biases__41', instrument_read(j__42,
                    'j__42'), None, None, False)
                write_instrument_read_sub(img__41[instrument_read(
                    instrument_read(i__41, 'i__41'), 'i__41')][
                    instrument_read(instrument_read(j__42, 'j__42'),
                    'j__42')],
                    "img__41[instrument_read(i__41, 'i__41')][instrument_read(j__42, 'j__42')]"
                    , instrument_read(instrument_read(k__43, 'k__43'),
                    'k__43'), None, None, False)
                print('exit scope 44')
            print('exit scope 43')
        print('exit scope 42')
    print('exit scope 41')
    return instrument_read(img__41, 'img__41')
    print('exit scope 41')


def flatten_layer(img):
    print('enter scope 45')
    print(1, 105)
    img__45 = instrument_read(img, 'img')
    write_instrument_read(img__45, 'img__45')
    print('malloc', sys.getsizeof(img__45), 'img__45')
    print(123, 106)
    result__45 = instrument_read(np, 'np').zeros(len(instrument_read(
        img__45, 'img__45')) * len(instrument_read_sub(instrument_read(
        img__45, 'img__45'), 'img__45', 0, None, None, False)) * len(
        instrument_read_sub(instrument_read_sub(instrument_read(img__45,
        'img__45'), 'img__45', 0, None, None, False), 'img__45[0]', 0, None,
        None, False)))
    write_instrument_read(result__45, 'result__45')
    print('malloc', sys.getsizeof(result__45), 'result__45')
    print(123, 107)
    index__45 = 0
    write_instrument_read(index__45, 'index__45')
    print('malloc', sys.getsizeof(index__45), 'index__45')
    for i__45 in range(len(instrument_read(img__45, 'img__45'))):
        print('enter scope 46')
        for j__46 in range(len(instrument_read_sub(instrument_read(img__45,
            'img__45'), 'img__45', 0, None, None, False))):
            print('enter scope 47')
            for k__47 in range(len(instrument_read_sub(instrument_read_sub(
                instrument_read(img__45, 'img__45'), 'img__45', 0, None,
                None, False), 'img__45[0]', 0, None, None, False))):
                print('enter scope 48')
                print(129, 111)
                result__45[instrument_read(instrument_read(index__45,
                    'index__45'), 'index__45')] = instrument_read_sub(
                    instrument_read_sub(instrument_read_sub(instrument_read
                    (img__45, 'img__45'), 'img__45', instrument_read(i__45,
                    'i__45'), None, None, False), 'img__45[i__45]',
                    instrument_read(j__46, 'j__46'), None, None, False),
                    'img__45[i__45][j__46]', instrument_read(k__47, 'k__47'
                    ), None, None, False)
                write_instrument_read_sub(result__45, 'result__45',
                    instrument_read(instrument_read(index__45, 'index__45'),
                    'index__45'), None, None, False)
                print(129, 112)
                index__45 += 1
                write_instrument_read(index__45, 'index__45')
                print('exit scope 48')
            print('exit scope 47')
        print('exit scope 46')
    print('exit scope 45')
    return instrument_read(result__45, 'result__45')
    print('exit scope 45')


def fc_layer(arr, W, W_0):
    print('enter scope 49')
    print(1, 115)
    arr__49 = instrument_read(arr, 'arr')
    write_instrument_read(arr__49, 'arr__49')
    print('malloc', sys.getsizeof(arr__49), 'arr__49')
    W__49 = instrument_read(W, 'W')
    write_instrument_read(W__49, 'W__49')
    print('malloc', sys.getsizeof(W__49), 'W__49')
    W_0__49 = instrument_read(W_0, 'W_0')
    write_instrument_read(W_0__49, 'W_0__49')
    print('malloc', sys.getsizeof(W_0__49), 'W_0__49')
    print(134, 116)
    result__49 = instrument_read(np, 'np').zeros(len(instrument_read_sub(
        instrument_read(W__49, 'W__49'), 'W__49', 0, None, None, False)))
    write_instrument_read(result__49, 'result__49')
    print('malloc', sys.getsizeof(result__49), 'result__49')
    for i__49 in range(len(instrument_read_sub(instrument_read(W__49,
        'W__49'), 'W__49', 0, None, None, False))):
        print('enter scope 50')
        print(136, 118)
        sum_val__50 = instrument_read_sub(instrument_read(W_0__49,
            'W_0__49'), 'W_0__49', instrument_read(i__49, 'i__49'), None,
            None, False)
        write_instrument_read(sum_val__50, 'sum_val__50')
        print('malloc', sys.getsizeof(sum_val__50), 'sum_val__50')
        for j__50 in range(len(instrument_read(arr__49, 'arr__49'))):
            print('enter scope 51')
            print(139, 120)
            sum_val__50 += instrument_read_sub(instrument_read(arr__49,
                'arr__49'), 'arr__49', instrument_read(j__50, 'j__50'),
                None, None, False) * instrument_read_sub(instrument_read_sub
                (instrument_read(W__49, 'W__49'), 'W__49', instrument_read(
                j__50, 'j__50'), None, None, False), 'W__49[j__50]',
                instrument_read(i__49, 'i__49'), None, None, False)
            write_instrument_read(sum_val__50, 'sum_val__50')
            print('exit scope 51')
        print(140, 121)
        result__49[instrument_read(instrument_read(i__49, 'i__49'), 'i__49')
            ] = instrument_read(sum_val__50, 'sum_val__50')
        write_instrument_read_sub(result__49, 'result__49', instrument_read
            (instrument_read(i__49, 'i__49'), 'i__49'), None, None, False)
        print('exit scope 50')
    print('exit scope 49')
    return instrument_read(result__49, 'result__49')
    print('exit scope 49')


def softmax(img):
    print('enter scope 52')
    print(1, 124)
    img__52 = instrument_read(img, 'img')
    write_instrument_read(img__52, 'img__52')
    print('malloc', sys.getsizeof(img__52), 'img__52')
    print(144, 125)
    sum_val__52 = 0
    write_instrument_read(sum_val__52, 'sum_val__52')
    print('malloc', sys.getsizeof(sum_val__52), 'sum_val__52')
    for i__52 in range(len(instrument_read(img__52, 'img__52'))):
        print('enter scope 53')
        print(146, 127)
        sum_val__52 += instrument_read(math, 'math').exp(instrument_read_sub
            (instrument_read(img__52, 'img__52'), 'img__52',
            instrument_read(i__52, 'i__52'), None, None, False))
        write_instrument_read(sum_val__52, 'sum_val__52')
        print('exit scope 53')
    print(147, 128)
    result__52 = instrument_read(np, 'np').zeros(len(instrument_read(
        img__52, 'img__52')))
    write_instrument_read(result__52, 'result__52')
    print('malloc', sys.getsizeof(result__52), 'result__52')
    for i__52 in range(len(instrument_read(img__52, 'img__52'))):
        print('enter scope 54')
        print(149, 130)
        result__52[instrument_read(instrument_read(i__52, 'i__52'), 'i__52')
            ] = instrument_read(math, 'math').exp(instrument_read_sub(
            instrument_read(img__52, 'img__52'), 'img__52', instrument_read
            (i__52, 'i__52'), None, None, False)) / instrument_read(sum_val__52
            , 'sum_val__52')
        write_instrument_read_sub(result__52, 'result__52', instrument_read
            (instrument_read(i__52, 'i__52'), 'i__52'), None, None, False)
        print('exit scope 54')
    print('exit scope 52')
    return instrument_read(result__52, 'result__52')
    print('exit scope 52')


def main():
    print('enter scope 55')
    print(1, 133)
    print(154, 134)
    zero_pad__55 = 3
    write_instrument_read(zero_pad__55, 'zero_pad__55')
    print('malloc', sys.getsizeof(zero_pad__55), 'zero_pad__55')
    print(154, 135)
    stride__55 = 2
    write_instrument_read(stride__55, 'stride__55')
    print('malloc', sys.getsizeof(stride__55), 'stride__55')
    print(154, 136)
    filt__55 = instrument_read(np, 'np').random.rand(3, 7, 7)
    write_instrument_read(filt__55, 'filt__55')
    print('malloc', sys.getsizeof(filt__55), 'filt__55')
    print(154, 137)
    img__55 = instrument_read(np, 'np').random.rand(3, 64, 64)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(154, 140)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, instrument_read(
        zero_pad__55, 'zero_pad__55'), instrument_read(stride__55,
        'stride__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(154, 141)
    weights__55 = instrument_read(np, 'np').random.rand(len(
        instrument_read_sub(instrument_read(img__55, 'img__55'), 'img__55',
        0, None, None, False)))
    write_instrument_read(weights__55, 'weights__55')
    print('malloc', sys.getsizeof(weights__55), 'weights__55')
    print(154, 142)
    biases__55 = instrument_read(np, 'np').random.rand(len(
        instrument_read_sub(instrument_read(img__55, 'img__55'), 'img__55',
        0, None, None, False)))
    write_instrument_read(biases__55, 'biases__55')
    print('malloc', sys.getsizeof(biases__55), 'biases__55')
    print(154, 143)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(154, 144)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(154, 145)
    img__55 = max_pool(instrument_read(img__55, 'img__55'), 3, 3, 1, 2)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(154, 148)
    filt__55 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False)), len(instrument_read_sub(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False), 'filt__55[0]', 0, None, None,
        False)))
    write_instrument_read(filt__55, 'filt__55')
    print('malloc', sys.getsizeof(filt__55), 'filt__55')
    for i__55 in range(2):
        print('enter scope 56')
        print(156, 150)
        byPass__56 = instrument_read(np, 'np').copy(instrument_read(img__55,
            'img__55'))
        write_instrument_read(byPass__56, 'byPass__56')
        print('malloc', sys.getsizeof(byPass__56), 'byPass__56')
        print(156, 151)
        img__55 = conv_layer(instrument_read(img__55, 'img__55'),
            instrument_read(filt__55, 'filt__55'), 1, 3, 1)
        write_instrument_read(img__55, 'img__55')
        print('malloc', sys.getsizeof(img__55), 'img__55')
        print(156, 152)
        img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
            instrument_read(weights__55, 'weights__55'), instrument_read(
            biases__55, 'biases__55'))
        write_instrument_read(img__55, 'img__55')
        print('malloc', sys.getsizeof(img__55), 'img__55')
        print(156, 153)
        img__55 = reLU(instrument_read(img__55, 'img__55'))
        write_instrument_read(img__55, 'img__55')
        print('malloc', sys.getsizeof(img__55), 'img__55')
        print(156, 154)
        img__55 = conv_layer(instrument_read(img__55, 'img__55'),
            instrument_read(filt__55, 'filt__55'), 1, 3, 1)
        write_instrument_read(img__55, 'img__55')
        print('malloc', sys.getsizeof(img__55), 'img__55')
        print(156, 155)
        img__55 += instrument_read(byPass__56, 'byPass__56')
        write_instrument_read(img__55, 'img__55')
        print(156, 156)
        img__55 = reLU(instrument_read(img__55, 'img__55'))
        write_instrument_read(img__55, 'img__55')
        print('malloc', sys.getsizeof(img__55), 'img__55')
        print('exit scope 56')
    print(157, 159)
    filt__55 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False)), len(instrument_read_sub(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False), 'filt__55[0]', 0, None, None,
        False)))
    write_instrument_read(filt__55, 'filt__55')
    print('malloc', sys.getsizeof(filt__55), 'filt__55')
    print(157, 160)
    byPass__55 = instrument_read(np, 'np').copy(instrument_read(img__55,
        'img__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 161)
    byPass__55 = conv_layer(instrument_read(byPass__55, 'byPass__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 2)
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 162)
    byPass__55 = BN_layer(instrument_read(byPass__55, 'byPass__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 163)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 164)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 165)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 166)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 2)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 167)
    img__55 += instrument_read(byPass__55, 'byPass__55')
    write_instrument_read(img__55, 'img__55')
    print(157, 168)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 170)
    byPass__55 = instrument_read(np, 'np').copy(instrument_read(img__55,
        'img__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 171)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 172)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 173)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 174)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 175)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 176)
    img__55 += instrument_read(byPass__55, 'byPass__55')
    write_instrument_read(img__55, 'img__55')
    print(157, 177)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 180)
    filt__55 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False)), len(instrument_read_sub(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False), 'filt__55[0]', 0, None, None,
        False)))
    write_instrument_read(filt__55, 'filt__55')
    print('malloc', sys.getsizeof(filt__55), 'filt__55')
    print(157, 181)
    byPass__55 = instrument_read(np, 'np').copy(instrument_read(img__55,
        'img__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 182)
    byPass__55 = conv_layer(instrument_read(byPass__55, 'byPass__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 2)
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 183)
    byPass__55 = BN_layer(instrument_read(byPass__55, 'byPass__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 184)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 185)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 186)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 187)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 2)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 188)
    img__55 += instrument_read(byPass__55, 'byPass__55')
    write_instrument_read(img__55, 'img__55')
    print(157, 189)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 191)
    byPass__55 = instrument_read(np, 'np').copy(instrument_read(img__55,
        'img__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 192)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 193)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 194)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 195)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 196)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 197)
    img__55 += instrument_read(byPass__55, 'byPass__55')
    write_instrument_read(img__55, 'img__55')
    print(157, 198)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 201)
    filt__55 = instrument_read(np, 'np').random.rand(1, len(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False)), len(instrument_read_sub(
        instrument_read_sub(instrument_read(filt__55, 'filt__55'),
        'filt__55', 0, None, None, False), 'filt__55[0]', 0, None, None,
        False)))
    write_instrument_read(filt__55, 'filt__55')
    print('malloc', sys.getsizeof(filt__55), 'filt__55')
    print(157, 202)
    byPass__55 = instrument_read(np, 'np').copy(instrument_read(img__55,
        'img__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 203)
    byPass__55 = conv_layer(instrument_read(byPass__55, 'byPass__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 2)
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 204)
    byPass__55 = BN_layer(instrument_read(byPass__55, 'byPass__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 205)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 206)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 207)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 208)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 2)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 209)
    img__55 += instrument_read(byPass__55, 'byPass__55')
    write_instrument_read(img__55, 'img__55')
    print(157, 210)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 212)
    byPass__55 = instrument_read(np, 'np').copy(instrument_read(img__55,
        'img__55'))
    write_instrument_read(byPass__55, 'byPass__55')
    print('malloc', sys.getsizeof(byPass__55), 'byPass__55')
    print(157, 213)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 214)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 215)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 216)
    img__55 = conv_layer(instrument_read(img__55, 'img__55'),
        instrument_read(filt__55, 'filt__55'), 1, 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 217)
    img__55 = BN_layer(instrument_read(img__55, 'img__55'), 1,
        instrument_read(weights__55, 'weights__55'), instrument_read(
        biases__55, 'biases__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 218)
    img__55 += instrument_read(byPass__55, 'byPass__55')
    write_instrument_read(img__55, 'img__55')
    print(157, 219)
    img__55 = reLU(instrument_read(img__55, 'img__55'))
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 220)
    img__55 = avg_pool(instrument_read(img__55, 'img__55'), len(
        instrument_read_sub(instrument_read(img__55, 'img__55'), 'img__55',
        0, None, None, False)), len(instrument_read_sub(instrument_read_sub
        (instrument_read(img__55, 'img__55'), 'img__55', 0, None, None,
        False), 'img__55[0]', 0, None, None, False)), 3, 1)
    write_instrument_read(img__55, 'img__55')
    print('malloc', sys.getsizeof(img__55), 'img__55')
    print(157, 221)
    flat__55 = flatten_layer(instrument_read(img__55, 'img__55'))
    write_instrument_read(flat__55, 'flat__55')
    print('malloc', sys.getsizeof(flat__55), 'flat__55')
    print(157, 223)
    weights__55 = instrument_read(np, 'np').random.rand(len(instrument_read
        (img__55, 'img__55')) * len(instrument_read_sub(instrument_read(
        img__55, 'img__55'), 'img__55', 0, None, None, False)) * len(
        instrument_read_sub(instrument_read_sub(instrument_read(img__55,
        'img__55'), 'img__55', 0, None, None, False), 'img__55[0]', 0, None,
        None, False)), 7)
    write_instrument_read(weights__55, 'weights__55')
    print('malloc', sys.getsizeof(weights__55), 'weights__55')
    print(157, 224)
    w_0__55 = instrument_read(np, 'np').random.rand(7)
    write_instrument_read(w_0__55, 'w_0__55')
    print('malloc', sys.getsizeof(w_0__55), 'w_0__55')
    print(157, 225)
    flat__55 = fc_layer(instrument_read(flat__55, 'flat__55'),
        instrument_read(weights__55, 'weights__55'), instrument_read(
        w_0__55, 'w_0__55'))
    write_instrument_read(flat__55, 'flat__55')
    print('malloc', sys.getsizeof(flat__55), 'flat__55')
    print(157, 226)
    final__55 = softmax(instrument_read(flat__55, 'flat__55'))
    write_instrument_read(final__55, 'final__55')
    print('malloc', sys.getsizeof(final__55), 'final__55')
    print('exit scope 55')
    return 0
    print('exit scope 55')


if instrument_read(__name__, '__name__') == '__main__':
    instrument_read(loop, 'loop').start_unroll
    main()
