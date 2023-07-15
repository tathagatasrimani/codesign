import sys
from instrument_lib import *
import sys
from instrument_lib import *
import math
import time
import numpy as np


def rosenbrock(x):
    print('enter scope 1')
    print(1, 8)
    x_1 = instrument_read(x, 'x')
    write_instrument_read(x_1, 'x_1')
    print('malloc', sys.getsizeof(x_1), 'x_1')
    print('exit scope 1')
    return 100 * (instrument_read_sub(instrument_read(x_1, 'x_1'), 'x_1', 1,
        None, None, False) - instrument_read_sub(instrument_read(x_1, 'x_1'
        ), 'x_1', 0, None, None, False) ** 2) ** 2 + (1 -
        instrument_read_sub(instrument_read(x_1, 'x_1'), 'x_1', 0, None,
        None, False)) ** 2
    print('exit scope 1')


def grad_rosen(x):
    print('enter scope 2')
    print(1, 12)
    x_2 = instrument_read(x, 'x')
    write_instrument_read(x_2, 'x_2')
    print('malloc', sys.getsizeof(x_2), 'x_2')
    print('exit scope 2')
    return instrument_read(np, 'np').array([200 * (instrument_read_sub(
        instrument_read(x_2, 'x_2'), 'x_2', 1, None, None, False) - 
        instrument_read_sub(instrument_read(x_2, 'x_2'), 'x_2', 0, None,
        None, False) ** 2) * (-2 * instrument_read_sub(instrument_read(x_2,
        'x_2'), 'x_2', 0, None, None, False)) + 2 * (instrument_read_sub(
        instrument_read(x_2, 'x_2'), 'x_2', 0, None, None, False) - 1), 200 *
        (instrument_read_sub(instrument_read(x_2, 'x_2'), 'x_2', 1, None,
        None, False) - instrument_read_sub(instrument_read(x_2, 'x_2'),
        'x_2', 0, None, None, False) ** 2)])
    print('exit scope 2')


def hessian_rosen(x):
    print('enter scope 3')
    print(1, 16)
    x_3 = instrument_read(x, 'x')
    write_instrument_read(x_3, 'x_3')
    print('malloc', sys.getsizeof(x_3), 'x_3')
    print('exit scope 3')
    return instrument_read(np, 'np').array([[1200 * instrument_read_sub(
        instrument_read(x_3, 'x_3'), 'x_3', 0, None, None, False) ** 2 - 
        400 * instrument_read_sub(instrument_read(x_3, 'x_3'), 'x_3', 1,
        None, None, False) + 2, -400 * instrument_read_sub(instrument_read(
        x_3, 'x_3'), 'x_3', 0, None, None, False)], [-400 *
        instrument_read_sub(instrument_read(x_3, 'x_3'), 'x_3', 0, None,
        None, False), 200]])
    print('exit scope 3')


def wolfe(f, g, xk, alpha, pk):
    print('enter scope 4')
    print(1, 21)
    f_4 = instrument_read(f, 'f')
    write_instrument_read(f_4, 'f_4')
    print('malloc', sys.getsizeof(f_4), 'f_4')
    g_4 = instrument_read(g, 'g')
    write_instrument_read(g_4, 'g_4')
    print('malloc', sys.getsizeof(g_4), 'g_4')
    xk_4 = instrument_read(xk, 'xk')
    write_instrument_read(xk_4, 'xk_4')
    print('malloc', sys.getsizeof(xk_4), 'xk_4')
    alpha_4 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha_4, 'alpha_4')
    print('malloc', sys.getsizeof(alpha_4), 'alpha_4')
    pk_4 = instrument_read(pk, 'pk')
    write_instrument_read(pk_4, 'pk_4')
    print('malloc', sys.getsizeof(pk_4), 'pk_4')
    print(15, 22)
    c1_4 = 0.0001
    write_instrument_read(c1_4, 'c1_4')
    print('malloc', sys.getsizeof(c1_4), 'c1_4')
    print('exit scope 4')
    return f(instrument_read(xk_4, 'xk_4') + instrument_read(alpha_4,
        'alpha_4') * instrument_read(pk_4, 'pk_4')) <= f(instrument_read(
        xk_4, 'xk_4')) + instrument_read(c1_4, 'c1_4') * instrument_read(
        alpha_4, 'alpha_4') * instrument_read(np, 'np').dot(g(
        instrument_read(xk_4, 'xk_4')), instrument_read(pk_4, 'pk_4'))
    print('exit scope 4')


def strong_wolfe(f, g, xk, alpha, pk, c2):
    print('enter scope 5')
    print(1, 26)
    f_5 = instrument_read(f, 'f')
    write_instrument_read(f_5, 'f_5')
    print('malloc', sys.getsizeof(f_5), 'f_5')
    g_5 = instrument_read(g, 'g')
    write_instrument_read(g_5, 'g_5')
    print('malloc', sys.getsizeof(g_5), 'g_5')
    xk_5 = instrument_read(xk, 'xk')
    write_instrument_read(xk_5, 'xk_5')
    print('malloc', sys.getsizeof(xk_5), 'xk_5')
    alpha_5 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha_5, 'alpha_5')
    print('malloc', sys.getsizeof(alpha_5), 'alpha_5')
    pk_5 = instrument_read(pk, 'pk')
    write_instrument_read(pk_5, 'pk_5')
    print('malloc', sys.getsizeof(pk_5), 'pk_5')
    c2_5 = instrument_read(c2, 'c2')
    write_instrument_read(c2_5, 'c2_5')
    print('malloc', sys.getsizeof(c2_5), 'c2_5')
    print('exit scope 5')
    return wolfe(instrument_read(f_5, 'f_5'), instrument_read(g_5, 'g_5'),
        instrument_read(xk_5, 'xk_5'), instrument_read(alpha_5, 'alpha_5'),
        instrument_read(pk_5, 'pk_5')) and abs(instrument_read(np, 'np').
        dot(g(instrument_read(xk_5, 'xk_5') + instrument_read(alpha_5,
        'alpha_5') * instrument_read(pk_5, 'pk_5')), instrument_read(pk_5,
        'pk_5'))) <= instrument_read(c2_5, 'c2_5') * abs(instrument_read(np,
        'np').dot(g(instrument_read(xk_5, 'xk_5')), instrument_read(pk_5,
        'pk_5')))
    print('exit scope 5')


def gold_stein(f, g, xk, alpha, pk, c):
    print('enter scope 6')
    print(1, 33)
    f_6 = instrument_read(f, 'f')
    write_instrument_read(f_6, 'f_6')
    print('malloc', sys.getsizeof(f_6), 'f_6')
    g_6 = instrument_read(g, 'g')
    write_instrument_read(g_6, 'g_6')
    print('malloc', sys.getsizeof(g_6), 'g_6')
    xk_6 = instrument_read(xk, 'xk')
    write_instrument_read(xk_6, 'xk_6')
    print('malloc', sys.getsizeof(xk_6), 'xk_6')
    alpha_6 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha_6, 'alpha_6')
    print('malloc', sys.getsizeof(alpha_6), 'alpha_6')
    pk_6 = instrument_read(pk, 'pk')
    write_instrument_read(pk_6, 'pk_6')
    print('malloc', sys.getsizeof(pk_6), 'pk_6')
    c_6 = instrument_read(c, 'c')
    write_instrument_read(c_6, 'c_6')
    print('malloc', sys.getsizeof(c_6), 'c_6')
    print('exit scope 6')
    return f(instrument_read(xk_6, 'xk_6')) + (1 - instrument_read(c_6, 'c_6')
        ) * instrument_read(alpha_6, 'alpha_6') * instrument_read(np, 'np'
        ).dot(g(instrument_read(xk_6, 'xk_6')), instrument_read(pk_6, 'pk_6')
        ) <= f(instrument_read(xk_6, 'xk_6') + instrument_read(alpha_6,
        'alpha_6') * instrument_read(pk_6, 'pk_6')) and f(instrument_read(
        xk_6, 'xk_6') + instrument_read(alpha_6, 'alpha_6') *
        instrument_read(pk_6, 'pk_6')) <= f(instrument_read(xk_6, 'xk_6')
        ) + instrument_read(c_6, 'c_6') * instrument_read(alpha_6, 'alpha_6'
        ) * instrument_read(np, 'np').dot(g(instrument_read(xk_6, 'xk_6')),
        instrument_read(pk_6, 'pk_6'))
    print('exit scope 6')


def step_length(f, g, xk, alpha, pk, c2):
    print('enter scope 7')
    print(1, 39)
    f_7 = instrument_read(f, 'f')
    write_instrument_read(f_7, 'f_7')
    print('malloc', sys.getsizeof(f_7), 'f_7')
    g_7 = instrument_read(g, 'g')
    write_instrument_read(g_7, 'g_7')
    print('malloc', sys.getsizeof(g_7), 'g_7')
    xk_7 = instrument_read(xk, 'xk')
    write_instrument_read(xk_7, 'xk_7')
    print('malloc', sys.getsizeof(xk_7), 'xk_7')
    alpha_7 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha_7, 'alpha_7')
    print('malloc', sys.getsizeof(alpha_7), 'alpha_7')
    pk_7 = instrument_read(pk, 'pk')
    write_instrument_read(pk_7, 'pk_7')
    print('malloc', sys.getsizeof(pk_7), 'pk_7')
    c2_7 = instrument_read(c2, 'c2')
    write_instrument_read(c2_7, 'c2_7')
    print('malloc', sys.getsizeof(c2_7), 'c2_7')
    print('exit scope 7')
    return interpolation(instrument_read(f_7, 'f_7'), instrument_read(g_7,
        'g_7'), lambda alpha_7: f(instrument_read(xk_7, 'xk_7') + 
        instrument_read(alpha_7, 'alpha_7') * instrument_read(pk_7, 'pk_7')
        ), lambda alpha_7: instrument_read(np, 'np').dot(g(instrument_read(
        xk_7, 'xk_7') + instrument_read(alpha_7, 'alpha_7') *
        instrument_read(pk_7, 'pk_7')), instrument_read(pk_7, 'pk_7')),
        instrument_read(alpha_7, 'alpha_7'), instrument_read(c2_7, 'c2_7'),
        lambda f_7, g_7, alpha_7, c2_7: strong_wolfe(instrument_read(f_7,
        'f_7'), instrument_read(g_7, 'g_7'), instrument_read(xk_7, 'xk_7'),
        instrument_read(alpha_7, 'alpha_7'), instrument_read(pk_7, 'pk_7'),
        instrument_read(c2_7, 'c2_7')))
    print('exit scope 7')


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha,
    iters=20):
    print('enter scope 8')
    print(1, 47)
    f_8 = instrument_read(f, 'f')
    write_instrument_read(f_8, 'f_8')
    print('malloc', sys.getsizeof(f_8), 'f_8')
    g_8 = instrument_read(g, 'g')
    write_instrument_read(g_8, 'g_8')
    print('malloc', sys.getsizeof(g_8), 'g_8')
    f_alpha_8 = instrument_read(f_alpha, 'f_alpha')
    write_instrument_read(f_alpha_8, 'f_alpha_8')
    print('malloc', sys.getsizeof(f_alpha_8), 'f_alpha_8')
    g_alpha_8 = instrument_read(g_alpha, 'g_alpha')
    write_instrument_read(g_alpha_8, 'g_alpha_8')
    print('malloc', sys.getsizeof(g_alpha_8), 'g_alpha_8')
    alpha_8 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha_8, 'alpha_8')
    print('malloc', sys.getsizeof(alpha_8), 'alpha_8')
    c2_8 = instrument_read(c2, 'c2')
    write_instrument_read(c2_8, 'c2_8')
    print('malloc', sys.getsizeof(c2_8), 'c2_8')
    strong_wolfe_alpha_8 = instrument_read(strong_wolfe_alpha,
        'strong_wolfe_alpha')
    write_instrument_read(strong_wolfe_alpha_8, 'strong_wolfe_alpha_8')
    print('malloc', sys.getsizeof(strong_wolfe_alpha_8), 'strong_wolfe_alpha_8'
        )
    iters_8 = instrument_read(iters, 'iters')
    write_instrument_read(iters_8, 'iters_8')
    print('malloc', sys.getsizeof(iters_8), 'iters_8')
    print(31, 50)
    l_8 = 0.0
    write_instrument_read(l_8, 'l_8')
    print('malloc', sys.getsizeof(l_8), 'l_8')
    print(31, 51)
    h_8 = 1.0
    write_instrument_read(h_8, 'h_8')
    print('malloc', sys.getsizeof(h_8), 'h_8')
    for i_8 in range(instrument_read(iters_8, 'iters_8')):
        if strong_wolfe_alpha(instrument_read(f_8, 'f_8'), instrument_read(
            g_8, 'g_8'), instrument_read(alpha_8, 'alpha_8'),
            instrument_read(c2_8, 'c2_8')):
            print('exit scope 8')
            return instrument_read(alpha_8, 'alpha_8')
        print(36, 56)
        half_8 = (instrument_read(l_8, 'l_8') + instrument_read(h_8, 'h_8')
            ) / 2
        write_instrument_read(half_8, 'half_8')
        print('malloc', sys.getsizeof(half_8), 'half_8')
        print(36, 57)
        alpha_8 = -g_alpha(instrument_read(l_8, 'l_8')) * instrument_read(h_8,
            'h_8') ** 2 / (2 * (f_alpha(instrument_read(h_8, 'h_8')) -
            f_alpha(instrument_read(l_8, 'l_8')) - g_alpha(instrument_read(
            l_8, 'l_8')) * instrument_read(h_8, 'h_8')))
        write_instrument_read(alpha_8, 'alpha_8')
        print('malloc', sys.getsizeof(alpha_8), 'alpha_8')
        if instrument_read(alpha_8, 'alpha_8') < instrument_read(l_8, 'l_8'
            ) or instrument_read(alpha_8, 'alpha_8') > instrument_read(h_8,
            'h_8'):
            print(38, 59)
            alpha_8 = instrument_read(half_8, 'half_8')
            write_instrument_read(alpha_8, 'alpha_8')
            print('malloc', sys.getsizeof(alpha_8), 'alpha_8')
        if g_alpha(instrument_read(alpha_8, 'alpha_8')) > 0:
            print(40, 61)
            h_8 = instrument_read(alpha_8, 'alpha_8')
            write_instrument_read(h_8, 'h_8')
            print('malloc', sys.getsizeof(h_8), 'h_8')
        elif g_alpha(instrument_read(alpha_8, 'alpha_8')) <= 0:
            print(43, 63)
            l_8 = instrument_read(alpha_8, 'alpha_8')
            write_instrument_read(l_8, 'l_8')
            print('malloc', sys.getsizeof(l_8), 'l_8')
    print('exit scope 8')
    return instrument_read(alpha_8, 'alpha_8')
    print('exit scope 8')


def conjugate_gradient(f, g, x0, iterations, error):
    print('enter scope 9')
    print(1, 68)
    f_9 = instrument_read(f, 'f')
    write_instrument_read(f_9, 'f_9')
    print('malloc', sys.getsizeof(f_9), 'f_9')
    g_9 = instrument_read(g, 'g')
    write_instrument_read(g_9, 'g_9')
    print('malloc', sys.getsizeof(g_9), 'g_9')
    x0_9 = instrument_read(x0, 'x0')
    write_instrument_read(x0_9, 'x0_9')
    print('malloc', sys.getsizeof(x0_9), 'x0_9')
    iterations_9 = instrument_read(iterations, 'iterations')
    write_instrument_read(iterations_9, 'iterations_9')
    print('malloc', sys.getsizeof(iterations_9), 'iterations_9')
    error_9 = instrument_read(error, 'error')
    write_instrument_read(error_9, 'error_9')
    print('malloc', sys.getsizeof(error_9), 'error_9')
    print(48, 69)
    xk_9 = instrument_read(x0_9, 'x0_9')
    write_instrument_read(xk_9, 'xk_9')
    print('malloc', sys.getsizeof(xk_9), 'xk_9')
    print(48, 70)
    c2_9 = 0.1
    write_instrument_read(c2_9, 'c2_9')
    print('malloc', sys.getsizeof(c2_9), 'c2_9')
    print(48, 72)
    fk_9 = f(instrument_read(xk_9, 'xk_9'))
    write_instrument_read(fk_9, 'fk_9')
    print('malloc', sys.getsizeof(fk_9), 'fk_9')
    print(48, 73)
    gk_9 = g(instrument_read(xk_9, 'xk_9'))
    write_instrument_read(gk_9, 'gk_9')
    print('malloc', sys.getsizeof(gk_9), 'gk_9')
    print(48, 74)
    pk_9 = -instrument_read(gk_9, 'gk_9')
    write_instrument_read(pk_9, 'pk_9')
    print('malloc', sys.getsizeof(pk_9), 'pk_9')
    for i_9 in range(instrument_read(iterations_9, 'iterations_9')):
        print(50, 77)
        alpha_9 = step_length(instrument_read(f_9, 'f_9'), instrument_read(
            g_9, 'g_9'), instrument_read(xk_9, 'xk_9'), 1.0,
            instrument_read(pk_9, 'pk_9'), instrument_read(c2_9, 'c2_9'))
        write_instrument_read(alpha_9, 'alpha_9')
        print('malloc', sys.getsizeof(alpha_9), 'alpha_9')
        print(50, 78)
        xk1_9 = instrument_read(xk_9, 'xk_9') + instrument_read(alpha_9,
            'alpha_9') * instrument_read(pk_9, 'pk_9')
        write_instrument_read(xk1_9, 'xk1_9')
        print('malloc', sys.getsizeof(xk1_9), 'xk1_9')
        print(50, 79)
        gk1_9 = g(instrument_read(xk1_9, 'xk1_9'))
        write_instrument_read(gk1_9, 'gk1_9')
        print('malloc', sys.getsizeof(gk1_9), 'gk1_9')
        print(50, 80)
        beta_k1_9 = instrument_read(np, 'np').dot(instrument_read(gk1_9,
            'gk1_9'), instrument_read(gk1_9, 'gk1_9')) / instrument_read(np,
            'np').dot(instrument_read(gk_9, 'gk_9'), instrument_read(gk_9,
            'gk_9'))
        write_instrument_read(beta_k1_9, 'beta_k1_9')
        print('malloc', sys.getsizeof(beta_k1_9), 'beta_k1_9')
        print(50, 81)
        pk1_9 = -instrument_read(gk1_9, 'gk1_9') + instrument_read(beta_k1_9,
            'beta_k1_9') * instrument_read(pk_9, 'pk_9')
        write_instrument_read(pk1_9, 'pk1_9')
        print('malloc', sys.getsizeof(pk1_9), 'pk1_9')
        if instrument_read(np, 'np').linalg.norm(instrument_read(xk1_9,
            'xk1_9') - instrument_read(xk_9, 'xk_9')) < instrument_read(error_9
            , 'error_9'):
            print(52, 84)
            xk_9 = instrument_read(xk1_9, 'xk1_9')
            write_instrument_read(xk_9, 'xk_9')
            print('malloc', sys.getsizeof(xk_9), 'xk_9')
            break
        print(53, 87)
        xk_9 = instrument_read(xk1_9, 'xk1_9')
        write_instrument_read(xk_9, 'xk_9')
        print('malloc', sys.getsizeof(xk_9), 'xk_9')
        print(53, 88)
        gk_9 = instrument_read(gk1_9, 'gk1_9')
        write_instrument_read(gk_9, 'gk_9')
        print('malloc', sys.getsizeof(gk_9), 'gk_9')
        print(53, 89)
        pk_9 = instrument_read(pk1_9, 'pk1_9')
        write_instrument_read(pk_9, 'pk_9')
        print('malloc', sys.getsizeof(pk_9), 'pk_9')
    print('exit scope 9')
    return instrument_read(xk_9, 'xk_9'), instrument_read(i_9, 'i_9') + 1
    print('exit scope 9')


if instrument_read(__name__, '__name__') == '__main__':
    print(56, 94)
    x0_0 = instrument_read(np, 'np').array([0, 0])
    write_instrument_read(x0_0, 'x0_0')
    print('malloc', sys.getsizeof(x0_0), 'x0_0')
    print(56, 95)
    error_0 = 0.0001
    write_instrument_read(error_0, 'error_0')
    print('malloc', sys.getsizeof(error_0), 'error_0')
    print(56, 96)
    max_iterations_0 = 1000
    write_instrument_read(max_iterations_0, 'max_iterations_0')
    print('malloc', sys.getsizeof(max_iterations_0), 'max_iterations_0')
    print(56, 99)
    start_0 = instrument_read(time, 'time').time()
    write_instrument_read(start_0, 'start_0')
    print('malloc', sys.getsizeof(start_0), 'start_0')
    print(56, 100)
    x_0, n_iter_0 = conjugate_gradient(instrument_read(rosenbrock,
        'rosenbrock'), instrument_read(grad_rosen, 'grad_rosen'),
        instrument_read(x0_0, 'x0_0'), iterations=max_iterations_0, error=
        error_0)
    write_instrument_read(n_iter_0, 'n_iter_0')
    print('malloc', sys.getsizeof(n_iter_0), 'n_iter_0')
    print(56, 102)
    end_0 = instrument_read(time, 'time').time()
    write_instrument_read(end_0, 'end_0')
    print('malloc', sys.getsizeof(end_0), 'end_0')
