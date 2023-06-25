import sys
import math
import time
import numpy as np


def rosenbrock(x):
    print(1, 8)
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosen(x):
    print(1, 12)
    return np.array([200 * (x[1] - x[0] ** 2) * (-2 * x[0]) + 2 * (x[0] - 1
        ), 200 * (x[1] - x[0] ** 2)])


def hessian_rosen(x):
    print(1, 16)
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-
        400 * x[0], 200]])


def wolfe(f, g, xk, alpha, pk):
    print(1, 21)
    print(15, 22)
    c1 = 0.0001
    return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


def strong_wolfe(f, g, xk, alpha, pk, c2):
    print(1, 26)
    return wolfe(f, g, xk, alpha, pk) and abs(np.dot(g(xk + alpha * pk), pk)
        ) <= c2 * abs(np.dot(g(xk), pk))


def gold_stein(f, g, xk, alpha, pk, c):
    print(1, 33)
    return f(xk) + (1 - c) * alpha * np.dot(g(xk), pk) <= f(xk + alpha * pk
        ) and f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk)


def step_length(f, g, xk, alpha, pk, c2):
    print(1, 39)
    return interpolation(f, g, lambda alpha: f(xk + alpha * pk), lambda
        alpha: np.dot(g(xk + alpha * pk), pk), alpha, c2, lambda f, g,
        alpha, c2: strong_wolfe(f, g, xk, alpha, pk, c2))


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha,
    iters=20):
    print(1, 47)
    print(31, 50)
    l = 0.0
    print(31, 51)
    h = 1.0
    print(32, 52)
    for i in range(iters):
        print(32, 52)
        if strong_wolfe_alpha(f, g, alpha, c2):
            print(33, 53)
            return alpha
        else:
            print(33, 53)
        print(36, 56)
        half = (l + h) / 2
        print(36, 57)
        alpha = -g_alpha(l) * h ** 2 / (2 * (f_alpha(h) - f_alpha(l) - 
            g_alpha(l) * h))
        if alpha < l or alpha > h:
            print(36, 58)
            print(38, 59)
            alpha = half
        else:
            print(36, 58)
        if g_alpha(alpha) > 0:
            print(39, 60)
            print(40, 61)
            h = alpha
        else:
            print(39, 60)
            if g_alpha(alpha) <= 0:
                print(42, 62)
                print(43, 63)
                l = alpha
            else:
                print(42, 62)
    return alpha


def conjugate_gradient(f, g, x0, iterations, error):
    print(1, 68)
    print(48, 69)
    xk = x0
    print(48, 70)
    c2 = 0.1
    print(48, 72)
    fk = f(xk)
    print(48, 73)
    gk = g(xk)
    print(48, 74)
    pk = -gk
    print(49, 76)
    for i in range(iterations):
        print(49, 76)
        print(50, 77)
        alpha = step_length(f, g, xk, 1.0, pk, c2)
        print(50, 78)
        xk1 = xk + alpha * pk
        print(50, 79)
        gk1 = g(xk1)
        print(50, 80)
        beta_k1 = np.dot(gk1, gk1) / np.dot(gk, gk)
        print(50, 81)
        pk1 = -gk1 + beta_k1 * pk
        if np.linalg.norm(xk1 - xk) < error:
            print(50, 83)
            print(52, 84)
            xk = xk1
            break
        else:
            print(50, 83)
        print(53, 87)
        xk = xk1
        print(53, 88)
        gk = gk1
        print(53, 89)
        pk = pk1
    return xk, i + 1


if __name__ == '__main__':
    print(1, 93)
    print(56, 94)
    x0 = np.array([0, 0])
    print(56, 95)
    error = 0.0001
    print(56, 96)
    max_iterations = 1000
    print(56, 99)
    start = time.time()
    print(56, 100)
    x, n_iter = conjugate_gradient(rosenbrock, grad_rosen, x0, iterations=
        max_iterations, error=error)
    print(56, 102)
    end = time.time()
else:
    print(1, 93)
