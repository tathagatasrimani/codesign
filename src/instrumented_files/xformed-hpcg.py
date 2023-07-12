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
    x__1 = instrument_read(x, 'x')
    write_instrument_read(x__1, 'x__1')
    print('malloc', sys.getsizeof(x__1), 'x__1')
    print('exit scope 1')
    return 100 * (instrument_read_sub(instrument_read(x__1, 'x__1'), 'x__1',
        1, None, None, False) - instrument_read_sub(instrument_read(x__1,
        'x__1'), 'x__1', 0, None, None, False) ** 2) ** 2 + (1 -
        instrument_read_sub(instrument_read(x__1, 'x__1'), 'x__1', 0, None,
        None, False)) ** 2
    print('exit scope 1')


def grad_rosen(x):
    print('enter scope 2')
    print(1, 12)
    x__2 = instrument_read(x, 'x')
    write_instrument_read(x__2, 'x__2')
    print('malloc', sys.getsizeof(x__2), 'x__2')
    print('exit scope 2')
    return instrument_read(np, 'np').array([200 * (instrument_read_sub(
        instrument_read(x__2, 'x__2'), 'x__2', 1, None, None, False) - 
        instrument_read_sub(instrument_read(x__2, 'x__2'), 'x__2', 0, None,
        None, False) ** 2) * (-2 * instrument_read_sub(instrument_read(x__2,
        'x__2'), 'x__2', 0, None, None, False)) + 2 * (instrument_read_sub(
        instrument_read(x__2, 'x__2'), 'x__2', 0, None, None, False) - 1), 
        200 * (instrument_read_sub(instrument_read(x__2, 'x__2'), 'x__2', 1,
        None, None, False) - instrument_read_sub(instrument_read(x__2,
        'x__2'), 'x__2', 0, None, None, False) ** 2)])
    print('exit scope 2')


def hessian_rosen(x):
    print('enter scope 3')
    print(1, 16)
    x__3 = instrument_read(x, 'x')
    write_instrument_read(x__3, 'x__3')
    print('malloc', sys.getsizeof(x__3), 'x__3')
    print('exit scope 3')
    return instrument_read(np, 'np').array([[1200 * instrument_read_sub(
        instrument_read(x__3, 'x__3'), 'x__3', 0, None, None, False) ** 2 -
        400 * instrument_read_sub(instrument_read(x__3, 'x__3'), 'x__3', 1,
        None, None, False) + 2, -400 * instrument_read_sub(instrument_read(
        x__3, 'x__3'), 'x__3', 0, None, None, False)], [-400 *
        instrument_read_sub(instrument_read(x__3, 'x__3'), 'x__3', 0, None,
        None, False), 200]])
    print('exit scope 3')


def wolfe(f, g, xk, alpha, pk):
    print('enter scope 4')
    print(1, 21)
    f__4 = instrument_read(f, 'f')
    write_instrument_read(f__4, 'f__4')
    print('malloc', sys.getsizeof(f__4), 'f__4')
    g__4 = instrument_read(g, 'g')
    write_instrument_read(g__4, 'g__4')
    print('malloc', sys.getsizeof(g__4), 'g__4')
    xk__4 = instrument_read(xk, 'xk')
    write_instrument_read(xk__4, 'xk__4')
    print('malloc', sys.getsizeof(xk__4), 'xk__4')
    alpha__4 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha__4, 'alpha__4')
    print('malloc', sys.getsizeof(alpha__4), 'alpha__4')
    pk__4 = instrument_read(pk, 'pk')
    write_instrument_read(pk__4, 'pk__4')
    print('malloc', sys.getsizeof(pk__4), 'pk__4')
    print(15, 22)
    c1__4 = 0.0001
    write_instrument_read(c1__4, 'c1__4')
    print('malloc', sys.getsizeof(c1__4), 'c1__4')
    print('exit scope 4')
    return f(instrument_read(xk__4, 'xk__4') + instrument_read(alpha__4,
        'alpha__4') * instrument_read(pk__4, 'pk__4')) <= f(instrument_read
        (xk__4, 'xk__4')) + instrument_read(c1__4, 'c1__4') * instrument_read(
        alpha__4, 'alpha__4') * instrument_read(np, 'np').dot(g(
        instrument_read(xk__4, 'xk__4')), instrument_read(pk__4, 'pk__4'))
    print('exit scope 4')


def strong_wolfe(f, g, xk, alpha, pk, c2):
    print('enter scope 5')
    print(1, 26)
    f__5 = instrument_read(f, 'f')
    write_instrument_read(f__5, 'f__5')
    print('malloc', sys.getsizeof(f__5), 'f__5')
    g__5 = instrument_read(g, 'g')
    write_instrument_read(g__5, 'g__5')
    print('malloc', sys.getsizeof(g__5), 'g__5')
    xk__5 = instrument_read(xk, 'xk')
    write_instrument_read(xk__5, 'xk__5')
    print('malloc', sys.getsizeof(xk__5), 'xk__5')
    alpha__5 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha__5, 'alpha__5')
    print('malloc', sys.getsizeof(alpha__5), 'alpha__5')
    pk__5 = instrument_read(pk, 'pk')
    write_instrument_read(pk__5, 'pk__5')
    print('malloc', sys.getsizeof(pk__5), 'pk__5')
    c2__5 = instrument_read(c2, 'c2')
    write_instrument_read(c2__5, 'c2__5')
    print('malloc', sys.getsizeof(c2__5), 'c2__5')
    print('exit scope 5')
    return wolfe(instrument_read(f__5, 'f__5'), instrument_read(g__5,
        'g__5'), instrument_read(xk__5, 'xk__5'), instrument_read(alpha__5,
        'alpha__5'), instrument_read(pk__5, 'pk__5')) and abs(instrument_read
        (np, 'np').dot(g(instrument_read(xk__5, 'xk__5') + instrument_read(
        alpha__5, 'alpha__5') * instrument_read(pk__5, 'pk__5')),
        instrument_read(pk__5, 'pk__5'))) <= instrument_read(c2__5, 'c2__5'
        ) * abs(instrument_read(np, 'np').dot(g(instrument_read(xk__5,
        'xk__5')), instrument_read(pk__5, 'pk__5')))
    print('exit scope 5')


def gold_stein(f, g, xk, alpha, pk, c):
    print('enter scope 6')
    print(1, 33)
    f__6 = instrument_read(f, 'f')
    write_instrument_read(f__6, 'f__6')
    print('malloc', sys.getsizeof(f__6), 'f__6')
    g__6 = instrument_read(g, 'g')
    write_instrument_read(g__6, 'g__6')
    print('malloc', sys.getsizeof(g__6), 'g__6')
    xk__6 = instrument_read(xk, 'xk')
    write_instrument_read(xk__6, 'xk__6')
    print('malloc', sys.getsizeof(xk__6), 'xk__6')
    alpha__6 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha__6, 'alpha__6')
    print('malloc', sys.getsizeof(alpha__6), 'alpha__6')
    pk__6 = instrument_read(pk, 'pk')
    write_instrument_read(pk__6, 'pk__6')
    print('malloc', sys.getsizeof(pk__6), 'pk__6')
    c__6 = instrument_read(c, 'c')
    write_instrument_read(c__6, 'c__6')
    print('malloc', sys.getsizeof(c__6), 'c__6')
    print('exit scope 6')
    return f(instrument_read(xk__6, 'xk__6')) + (1 - instrument_read(c__6,
        'c__6')) * instrument_read(alpha__6, 'alpha__6') * instrument_read(np,
        'np').dot(g(instrument_read(xk__6, 'xk__6')), instrument_read(pk__6,
        'pk__6')) <= f(instrument_read(xk__6, 'xk__6') + instrument_read(
        alpha__6, 'alpha__6') * instrument_read(pk__6, 'pk__6')) and f(
        instrument_read(xk__6, 'xk__6') + instrument_read(alpha__6,
        'alpha__6') * instrument_read(pk__6, 'pk__6')) <= f(instrument_read
        (xk__6, 'xk__6')) + instrument_read(c__6, 'c__6') * instrument_read(
        alpha__6, 'alpha__6') * instrument_read(np, 'np').dot(g(
        instrument_read(xk__6, 'xk__6')), instrument_read(pk__6, 'pk__6'))
    print('exit scope 6')


def step_length(f, g, xk, alpha, pk, c2):
    print('enter scope 7')
    print(1, 39)
    f__7 = instrument_read(f, 'f')
    write_instrument_read(f__7, 'f__7')
    print('malloc', sys.getsizeof(f__7), 'f__7')
    g__7 = instrument_read(g, 'g')
    write_instrument_read(g__7, 'g__7')
    print('malloc', sys.getsizeof(g__7), 'g__7')
    xk__7 = instrument_read(xk, 'xk')
    write_instrument_read(xk__7, 'xk__7')
    print('malloc', sys.getsizeof(xk__7), 'xk__7')
    alpha__7 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha__7, 'alpha__7')
    print('malloc', sys.getsizeof(alpha__7), 'alpha__7')
    pk__7 = instrument_read(pk, 'pk')
    write_instrument_read(pk__7, 'pk__7')
    print('malloc', sys.getsizeof(pk__7), 'pk__7')
    c2__7 = instrument_read(c2, 'c2')
    write_instrument_read(c2__7, 'c2__7')
    print('malloc', sys.getsizeof(c2__7), 'c2__7')
    print('exit scope 7')
    return interpolation(instrument_read(f__7, 'f__7'), instrument_read(
        g__7, 'g__7'), lambda alpha__7: f(instrument_read(xk__7, 'xk__7') +
        instrument_read(alpha__7, 'alpha__7') * instrument_read(pk__7,
        'pk__7')), lambda alpha__7: instrument_read(np, 'np').dot(g(
        instrument_read(xk__7, 'xk__7') + instrument_read(alpha__7,
        'alpha__7') * instrument_read(pk__7, 'pk__7')), instrument_read(
        pk__7, 'pk__7')), instrument_read(alpha__7, 'alpha__7'),
        instrument_read(c2__7, 'c2__7'), lambda f__7, g__7, alpha__7, c2__7:
        strong_wolfe(instrument_read(f__7, 'f__7'), instrument_read(g__7,
        'g__7'), instrument_read(xk__7, 'xk__7'), instrument_read(alpha__7,
        'alpha__7'), instrument_read(pk__7, 'pk__7'), instrument_read(c2__7,
        'c2__7')))
    print('exit scope 7')


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha,
    iters=20):
    print('enter scope 8')
    print(1, 47)
    f__8 = instrument_read(f, 'f')
    write_instrument_read(f__8, 'f__8')
    print('malloc', sys.getsizeof(f__8), 'f__8')
    g__8 = instrument_read(g, 'g')
    write_instrument_read(g__8, 'g__8')
    print('malloc', sys.getsizeof(g__8), 'g__8')
    f_alpha__8 = instrument_read(f_alpha, 'f_alpha')
    write_instrument_read(f_alpha__8, 'f_alpha__8')
    print('malloc', sys.getsizeof(f_alpha__8), 'f_alpha__8')
    g_alpha__8 = instrument_read(g_alpha, 'g_alpha')
    write_instrument_read(g_alpha__8, 'g_alpha__8')
    print('malloc', sys.getsizeof(g_alpha__8), 'g_alpha__8')
    alpha__8 = instrument_read(alpha, 'alpha')
    write_instrument_read(alpha__8, 'alpha__8')
    print('malloc', sys.getsizeof(alpha__8), 'alpha__8')
    c2__8 = instrument_read(c2, 'c2')
    write_instrument_read(c2__8, 'c2__8')
    print('malloc', sys.getsizeof(c2__8), 'c2__8')
    strong_wolfe_alpha__8 = instrument_read(strong_wolfe_alpha,
        'strong_wolfe_alpha')
    write_instrument_read(strong_wolfe_alpha__8, 'strong_wolfe_alpha__8')
    print('malloc', sys.getsizeof(strong_wolfe_alpha__8),
        'strong_wolfe_alpha__8')
    iters__8 = instrument_read(iters, 'iters')
    write_instrument_read(iters__8, 'iters__8')
    print('malloc', sys.getsizeof(iters__8), 'iters__8')
    print(31, 50)
    l__8 = 0.0
    write_instrument_read(l__8, 'l__8')
    print('malloc', sys.getsizeof(l__8), 'l__8')
    print(31, 51)
    h__8 = 1.0
    write_instrument_read(h__8, 'h__8')
    print('malloc', sys.getsizeof(h__8), 'h__8')
    for i__8 in range(instrument_read(iters__8, 'iters__8')):
        if strong_wolfe_alpha(instrument_read(f__8, 'f__8'),
            instrument_read(g__8, 'g__8'), instrument_read(alpha__8,
            'alpha__8'), instrument_read(c2__8, 'c2__8')):
            print('exit scope 8')
            return instrument_read(alpha__8, 'alpha__8')
        print(36, 56)
        half__8 = (instrument_read(l__8, 'l__8') + instrument_read(h__8,
            'h__8')) / 2
        write_instrument_read(half__8, 'half__8')
        print('malloc', sys.getsizeof(half__8), 'half__8')
        print(36, 57)
        alpha__8 = -g_alpha(instrument_read(l__8, 'l__8')) * instrument_read(
            h__8, 'h__8') ** 2 / (2 * (f_alpha(instrument_read(h__8, 'h__8'
            )) - f_alpha(instrument_read(l__8, 'l__8')) - g_alpha(
            instrument_read(l__8, 'l__8')) * instrument_read(h__8, 'h__8')))
        write_instrument_read(alpha__8, 'alpha__8')
        print('malloc', sys.getsizeof(alpha__8), 'alpha__8')
        if instrument_read(alpha__8, 'alpha__8') < instrument_read(l__8, 'l__8'
            ) or instrument_read(alpha__8, 'alpha__8') > instrument_read(h__8,
            'h__8'):
            print(38, 59)
            alpha__8 = instrument_read(half__8, 'half__8')
            write_instrument_read(alpha__8, 'alpha__8')
            print('malloc', sys.getsizeof(alpha__8), 'alpha__8')
        if g_alpha(instrument_read(alpha__8, 'alpha__8')) > 0:
            print(40, 61)
            h__8 = instrument_read(alpha__8, 'alpha__8')
            write_instrument_read(h__8, 'h__8')
            print('malloc', sys.getsizeof(h__8), 'h__8')
        elif g_alpha(instrument_read(alpha__8, 'alpha__8')) <= 0:
            print(43, 63)
            l__8 = instrument_read(alpha__8, 'alpha__8')
            write_instrument_read(l__8, 'l__8')
            print('malloc', sys.getsizeof(l__8), 'l__8')
    print('exit scope 8')
    return instrument_read(alpha__8, 'alpha__8')
    print('exit scope 8')


def conjugate_gradient(f, g, x0, iterations, error):
    print('enter scope 9')
    print(1, 68)
    f__9 = instrument_read(f, 'f')
    write_instrument_read(f__9, 'f__9')
    print('malloc', sys.getsizeof(f__9), 'f__9')
    g__9 = instrument_read(g, 'g')
    write_instrument_read(g__9, 'g__9')
    print('malloc', sys.getsizeof(g__9), 'g__9')
    x0__9 = instrument_read(x0, 'x0')
    write_instrument_read(x0__9, 'x0__9')
    print('malloc', sys.getsizeof(x0__9), 'x0__9')
    iterations__9 = instrument_read(iterations, 'iterations')
    write_instrument_read(iterations__9, 'iterations__9')
    print('malloc', sys.getsizeof(iterations__9), 'iterations__9')
    error__9 = instrument_read(error, 'error')
    write_instrument_read(error__9, 'error__9')
    print('malloc', sys.getsizeof(error__9), 'error__9')
    print(48, 69)
    xk__9 = instrument_read(x0__9, 'x0__9')
    write_instrument_read(xk__9, 'xk__9')
    print('malloc', sys.getsizeof(xk__9), 'xk__9')
    print(48, 70)
    c2__9 = 0.1
    write_instrument_read(c2__9, 'c2__9')
    print('malloc', sys.getsizeof(c2__9), 'c2__9')
    print(48, 72)
    fk__9 = f(instrument_read(xk__9, 'xk__9'))
    write_instrument_read(fk__9, 'fk__9')
    print('malloc', sys.getsizeof(fk__9), 'fk__9')
    print(48, 73)
    gk__9 = g(instrument_read(xk__9, 'xk__9'))
    write_instrument_read(gk__9, 'gk__9')
    print('malloc', sys.getsizeof(gk__9), 'gk__9')
    print(48, 74)
    pk__9 = -instrument_read(gk__9, 'gk__9')
    write_instrument_read(pk__9, 'pk__9')
    print('malloc', sys.getsizeof(pk__9), 'pk__9')
    for i__9 in range(instrument_read(iterations__9, 'iterations__9')):
        print(50, 77)
        alpha__9 = step_length(instrument_read(f__9, 'f__9'),
            instrument_read(g__9, 'g__9'), instrument_read(xk__9, 'xk__9'),
            1.0, instrument_read(pk__9, 'pk__9'), instrument_read(c2__9,
            'c2__9'))
        write_instrument_read(alpha__9, 'alpha__9')
        print('malloc', sys.getsizeof(alpha__9), 'alpha__9')
        print(50, 78)
        xk1__9 = instrument_read(xk__9, 'xk__9') + instrument_read(alpha__9,
            'alpha__9') * instrument_read(pk__9, 'pk__9')
        write_instrument_read(xk1__9, 'xk1__9')
        print('malloc', sys.getsizeof(xk1__9), 'xk1__9')
        print(50, 79)
        gk1__9 = g(instrument_read(xk1__9, 'xk1__9'))
        write_instrument_read(gk1__9, 'gk1__9')
        print('malloc', sys.getsizeof(gk1__9), 'gk1__9')
        print(50, 80)
        beta_k1__9 = instrument_read(np, 'np').dot(instrument_read(gk1__9,
            'gk1__9'), instrument_read(gk1__9, 'gk1__9')) / instrument_read(np,
            'np').dot(instrument_read(gk__9, 'gk__9'), instrument_read(
            gk__9, 'gk__9'))
        write_instrument_read(beta_k1__9, 'beta_k1__9')
        print('malloc', sys.getsizeof(beta_k1__9), 'beta_k1__9')
        print(50, 81)
        pk1__9 = -instrument_read(gk1__9, 'gk1__9') + instrument_read(
            beta_k1__9, 'beta_k1__9') * instrument_read(pk__9, 'pk__9')
        write_instrument_read(pk1__9, 'pk1__9')
        print('malloc', sys.getsizeof(pk1__9), 'pk1__9')
        if instrument_read(np, 'np').linalg.norm(instrument_read(xk1__9,
            'xk1__9') - instrument_read(xk__9, 'xk__9')) < instrument_read(
            error__9, 'error__9'):
            print(52, 84)
            xk__9 = instrument_read(xk1__9, 'xk1__9')
            write_instrument_read(xk__9, 'xk__9')
            print('malloc', sys.getsizeof(xk__9), 'xk__9')
            break
        print(53, 87)
        xk__9 = instrument_read(xk1__9, 'xk1__9')
        write_instrument_read(xk__9, 'xk__9')
        print('malloc', sys.getsizeof(xk__9), 'xk__9')
        print(53, 88)
        gk__9 = instrument_read(gk1__9, 'gk1__9')
        write_instrument_read(gk__9, 'gk__9')
        print('malloc', sys.getsizeof(gk__9), 'gk__9')
        print(53, 89)
        pk__9 = instrument_read(pk1__9, 'pk1__9')
        write_instrument_read(pk__9, 'pk__9')
        print('malloc', sys.getsizeof(pk__9), 'pk__9')
    print('exit scope 9')
    return instrument_read(xk__9, 'xk__9'), instrument_read(i__9, 'i__9') + 1
    print('exit scope 9')


if instrument_read(__name__, '__name__') == '__main__':
    print(56, 94)
    x0__0 = instrument_read(np, 'np').array([0, 0])
    write_instrument_read(x0__0, 'x0__0')
    print('malloc', sys.getsizeof(x0__0), 'x0__0')
    print(56, 95)
    error__0 = 0.0001
    write_instrument_read(error__0, 'error__0')
    print('malloc', sys.getsizeof(error__0), 'error__0')
    print(56, 96)
    max_iterations__0 = 1000
    write_instrument_read(max_iterations__0, 'max_iterations__0')
    print('malloc', sys.getsizeof(max_iterations__0), 'max_iterations__0')
    print(56, 99)
    start__0 = instrument_read(time, 'time').time()
    write_instrument_read(start__0, 'start__0')
    print('malloc', sys.getsizeof(start__0), 'start__0')
    print(56, 100)
    x__0, n_iter__0 = conjugate_gradient(instrument_read(rosenbrock,
        'rosenbrock'), instrument_read(grad_rosen, 'grad_rosen'),
        instrument_read(x0__0, 'x0__0'), iterations=max_iterations__0,
        error=error__0)
    write_instrument_read(n_iter__0, 'n_iter__0')
    print('malloc', sys.getsizeof(n_iter__0), 'n_iter__0')
    print(56, 102)
    end__0 = instrument_read(time, 'time').time()
    write_instrument_read(end__0, 'end__0')
    print('malloc', sys.getsizeof(end__0), 'end__0')
