import sys
from instrument_lib import *
import math
import time
import numpy as np


def rosenbrock(x):
    print('enter scope 1')
    print(1, 8)
    x__1 = x
    print('exit scope 1')
    return 100 * (x__1[1] - x__1[0] ** 2) ** 2 + (1 - x__1[0]) ** 2
    print('exit scope 1')


def grad_rosen(x):
    print('enter scope 2')
    print(1, 12)
    x__2 = x
    print('exit scope 2')
    return np.array([200 * (x__2[1] - x__2[0] ** 2) * (-2 * x__2[0]) + 2 *
        (x__2[0] - 1), 200 * (x__2[1] - x__2[0] ** 2)])
    print('exit scope 2')


def hessian_rosen(x):
    print('enter scope 3')
    print(1, 16)
    x__3 = x
    print('exit scope 3')
    return np.array([[1200 * x__3[0] ** 2 - 400 * x__3[1] + 2, -400 * x__3[
        0]], [-400 * x__3[0], 200]])
    print('exit scope 3')


def wolfe(f, g, xk, alpha, pk):
    print('enter scope 4')
    print(1, 21)
    f__4 = f
    g__4 = g
    xk__4 = xk
    alpha__4 = alpha
    pk__4 = pk
    print(15, 22)
    c1__4 = 0.0001
    print('exit scope 4')
    return f(xk__4 + alpha__4 * pk__4) <= f(xk__4) + c1__4 * alpha__4 * np.dot(
        g(xk__4), pk__4)
    print('exit scope 4')


def strong_wolfe(f, g, xk, alpha, pk, c2):
    print('enter scope 5')
    print(1, 26)
    f__5 = f
    g__5 = g
    xk__5 = xk
    alpha__5 = alpha
    pk__5 = pk
    c2__5 = c2
    print('exit scope 5')
    return wolfe(f__5, g__5, xk__5, alpha__5, pk__5) and abs(np.dot(g(xk__5 +
        alpha__5 * pk__5), pk__5)) <= c2__5 * abs(np.dot(g(xk__5), pk__5))
    print('exit scope 5')


def gold_stein(f, g, xk, alpha, pk, c):
    print('enter scope 6')
    print(1, 33)
    f__6 = f
    g__6 = g
    xk__6 = xk
    alpha__6 = alpha
    pk__6 = pk
    c__6 = c
    print('exit scope 6')
    return f(xk__6) + (1 - c__6) * alpha__6 * np.dot(g(xk__6), pk__6) <= f(
        xk__6 + alpha__6 * pk__6) and f(xk__6 + alpha__6 * pk__6) <= f(xk__6
        ) + c__6 * alpha__6 * np.dot(g(xk__6), pk__6)
    print('exit scope 6')


def step_length(f, g, xk, alpha, pk, c2):
    print('enter scope 7')
    print(1, 39)
    f__7 = f
    g__7 = g
    xk__7 = xk
    alpha__7 = alpha
    pk__7 = pk
    c2__7 = c2
    print('exit scope 7')
    return interpolation(f__7, g__7, lambda alpha__7: f(xk__7 + alpha__7 *
        pk__7), lambda alpha__7: np.dot(g(xk__7 + alpha__7 * pk__7), pk__7),
        alpha__7, c2__7, lambda f__7, g__7, alpha__7, c2__7: strong_wolfe(
        f__7, g__7, xk__7, alpha__7, pk__7, c2__7))
    print('exit scope 7')


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha,
    iters=20):
    print('enter scope 8')
    print(1, 47)
    f__8 = f
    g__8 = g
    f_alpha__8 = f_alpha
    g_alpha__8 = g_alpha
    alpha__8 = alpha
    c2__8 = c2
    strong_wolfe_alpha__8 = strong_wolfe_alpha
    iters__8 = iters
    print(31, 50)
    l__8 = 0.0
    print(31, 51)
    h__8 = 1.0
    for i__8 in range(iters__8):
        if strong_wolfe_alpha(f__8, g__8, alpha__8, c2__8):
            print('exit scope 8')
            return alpha__8
        print(36, 56)
        half__8 = (l__8 + h__8) / 2
        print(36, 57)
        alpha__8 = -g_alpha(l__8) * h__8 ** 2 / (2 * (f_alpha(h__8) -
            f_alpha(l__8) - g_alpha(l__8) * h__8))
        if alpha__8 < l__8 or alpha__8 > h__8:
            print(38, 59)
            alpha__8 = half__8
        if g_alpha(alpha__8) > 0:
            print(40, 61)
            h__8 = alpha__8
        elif g_alpha(alpha__8) <= 0:
            print(43, 63)
            l__8 = alpha__8
    print('exit scope 8')
    return alpha__8
    print('exit scope 8')


def conjugate_gradient(f, g, x0, iterations, error):
    print('enter scope 9')
    print(1, 68)
    f__9 = f
    g__9 = g
    x0__9 = x0
    iterations__9 = iterations
    error__9 = error
    print(48, 69)
    xk__9 = x0__9
    print(48, 70)
    c2__9 = 0.1
    print(48, 72)
    fk__9 = f(xk__9)
    print(48, 73)
    gk__9 = g(xk__9)
    print(48, 74)
    pk__9 = -gk__9
    for i__9 in range(iterations__9):
        print(50, 77)
        alpha__9 = step_length(f__9, g__9, xk__9, 1.0, pk__9, c2__9)
        print(50, 78)
        xk1__9 = xk__9 + alpha__9 * pk__9
        print(50, 79)
        gk1__9 = g(xk1__9)
        print(50, 80)
        beta_k1__9 = np.dot(gk1__9, gk1__9) / np.dot(gk__9, gk__9)
        print(50, 81)
        pk1__9 = -gk1__9 + beta_k1__9 * pk__9
        if np.linalg.norm(xk1__9 - xk__9) < error__9:
            print(52, 84)
            xk__9 = xk1__9
            break
        print(53, 87)
        xk__9 = xk1__9
        print(53, 88)
        gk__9 = gk1__9
        print(53, 89)
        pk__9 = pk1__9
    print('exit scope 9')
    return xk__9, i__9 + 1
    print('exit scope 9')


if __name__ == '__main__':
    print(56, 94)
    x0__0 = np.array([0, 0])
    print(56, 95)
    error__0 = 0.0001
    print(56, 96)
    max_iterations__0 = 1000
    print(56, 99)
    start__0 = time.time()
    print(56, 100)
    x__0, n_iter__0 = conjugate_gradient(rosenbrock, grad_rosen, x0__0,
        iterations=max_iterations__0, error=error__0)
    print(56, 102)
    end__0 = time.time()
