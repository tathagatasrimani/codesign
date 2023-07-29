import sys
from instrument_lib import *
import math
import time
import numpy as np


def rosenbrock(x):
    print('enter scope 1')
    print(1, 8)
    x_1 = x
    print('exit scope 1')
    return 100 * (x_1[1] - x_1[0] ** 2) ** 2 + (1 - x_1[0]) ** 2
    print('exit scope 1')


def grad_rosen(x):
    print('enter scope 2')
    print(1, 12)
    x_2 = x
    print(7, 13)
    arr_2 = np.array([200 * (x_2[1] - x_2[0] ** 2) * (-2 * x_2[0]) + 2 * (
        x_2[0] - 1), 200 * (x_2[1] - x_2[0] ** 2)])
    print(7, 14)
    arr_2 = np.repeat(arr_2, 2048)
    print('exit scope 2')
    return arr_2
    print('exit scope 2')


def hessian_rosen(x):
    print('enter scope 3')
    print(1, 18)
    x_3 = x
    print('exit scope 3')
    return np.array([[1200 * x_3[0] ** 2 - 400 * x_3[1] + 2, -400 * x_3[0]],
        [-400 * x_3[0], 200]])
    print('exit scope 3')


def wolfe(f, g, xk, alpha, pk):
    print('enter scope 4')
    print(1, 23)
    f_4 = f
    g_4 = g
    xk_4 = xk
    alpha_4 = alpha
    pk_4 = pk
    print(15, 24)
    c1_4 = 0.0001
    print('exit scope 4')
    return f(xk_4 + alpha_4 * pk_4) <= f(xk_4) + c1_4 * alpha_4 * np.dot(g(
        xk_4), pk_4)
    print('exit scope 4')


def strong_wolfe(f, g, xk, alpha, pk, c2):
    print('enter scope 5')
    print(1, 28)
    f_5 = f
    g_5 = g
    xk_5 = xk
    alpha_5 = alpha
    pk_5 = pk
    c2_5 = c2
    print('exit scope 5')
    return wolfe(f_5, g_5, xk_5, alpha_5, pk_5) and abs(np.dot(g(xk_5 + 
        alpha_5 * pk_5), pk_5)) <= c2_5 * abs(np.dot(g(xk_5), pk_5))
    print('exit scope 5')


def gold_stein(f, g, xk, alpha, pk, c):
    print('enter scope 6')
    print(1, 35)
    f_6 = f
    g_6 = g
    xk_6 = xk
    alpha_6 = alpha
    pk_6 = pk
    c_6 = c
    print('exit scope 6')
    return f(xk_6) + (1 - c_6) * alpha_6 * np.dot(g(xk_6), pk_6) <= f(xk_6 +
        alpha_6 * pk_6) and f(xk_6 + alpha_6 * pk_6) <= f(xk_6
        ) + c_6 * alpha_6 * np.dot(g(xk_6), pk_6)
    print('exit scope 6')


def step_length(f, g, xk, alpha, pk, c2):
    print('enter scope 7')
    print(1, 41)
    f_7 = f
    g_7 = g
    xk_7 = xk
    alpha_7 = alpha
    pk_7 = pk
    c2_7 = c2
    print('exit scope 7')
    return interpolation(f_7, g_7, lambda alpha_7: f(xk_7 + alpha_7 * pk_7),
        lambda alpha_7: np.dot(g(xk_7 + alpha_7 * pk_7), pk_7), alpha_7,
        c2_7, lambda f_7, g_7, alpha_7, c2_7: strong_wolfe(f_7, g_7, xk_7,
        alpha_7, pk_7, c2_7))
    print('exit scope 7')


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha,
    iters=20):
    print('enter scope 8')
    print(1, 49)
    f_8 = f
    g_8 = g
    f_alpha_8 = f_alpha
    g_alpha_8 = g_alpha
    alpha_8 = alpha
    c2_8 = c2
    strong_wolfe_alpha_8 = strong_wolfe_alpha
    iters_8 = iters
    print(31, 52)
    l_8 = 0.0
    print(31, 53)
    h_8 = 1.0
    for i_8 in range(iters_8):
        if strong_wolfe_alpha(f_8, g_8, alpha_8, c2_8):
            print('exit scope 8')
            return alpha_8
        print(36, 58)
        half_8 = (l_8 + h_8) / 2
        print(36, 59)
        alpha_8 = -g_alpha(l_8) * h_8 ** 2 / (2 * (f_alpha(h_8) - f_alpha(
            l_8) - g_alpha(l_8) * h_8))
        if alpha_8 < l_8 or alpha_8 > h_8:
            print(38, 61)
            alpha_8 = half_8
        if g_alpha(alpha_8) > 0:
            print(40, 63)
            h_8 = alpha_8
        elif g_alpha(alpha_8) <= 0:
            print(43, 65)
            l_8 = alpha_8
    print('exit scope 8')
    return alpha_8
    print('exit scope 8')


def conjugate_gradient(f, g, x0, iterations, error):
    print('enter scope 9')
    print(1, 70)
    f_9 = f
    g_9 = g
    x0_9 = x0
    iterations_9 = iterations
    error_9 = error
    print(48, 71)
    xk_9 = x0_9
    print(48, 72)
    c2_9 = 0.1
    print(48, 74)
    fk_9 = f(xk_9)
    print(48, 75)
    gk_9 = g(xk_9)
    print(48, 76)
    pk_9 = -gk_9
    for i_9 in range(iterations_9):
        print(50, 79)
        alpha_9 = step_length(f_9, g_9, xk_9, 1.0, pk_9, c2_9)
        print(50, 80)
        xk1_9 = xk_9 + alpha_9 * pk_9
        print(50, 81)
        gk1_9 = g(xk1_9)
        print(50, 82)
        beta_k1_9 = np.dot(gk1_9, gk1_9) / np.dot(gk_9, gk_9)
        print(50, 83)
        pk1_9 = -gk1_9 + beta_k1_9 * pk_9
        if np.linalg.norm(xk1_9 - xk_9) < error_9:
            print(52, 86)
            xk_9 = xk1_9
            break
        print(53, 89)
        xk_9 = xk1_9
        print(53, 90)
        gk_9 = gk1_9
        print(53, 91)
        pk_9 = pk1_9
    print('exit scope 9')
    return xk_9, i_9 + 1
    print('exit scope 9')


if __name__ == '__main__':
    print(56, 95)
    x0_0 = np.zeros([4096])
    print(56, 96)
    error_0 = 0.0001
    print(56, 97)
    max_iterations_0 = 10000
    print(56, 100)
    start_0 = time.time()
    print(56, 101)
    x_0, n_iter_0 = conjugate_gradient(rosenbrock, grad_rosen, x0_0,
        iterations=max_iterations_0, error=error_0)
    print(56, 103)
    end_0 = time.time()
