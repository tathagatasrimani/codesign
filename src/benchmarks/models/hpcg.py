import math
import time
import numpy as np


# 2d rosenbrock function and its first and second order derivatives
#     https://en.wikipedia.org/wiki/Rosenbrock_function
def rosenbrock(x):
  return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def grad_rosen(x):
  arr = np.array([200*(x[1]-x[0]**2)*(-2*x[0]) + 2*(x[0]-1), 200*(x[1]-x[0]**2)])
  arr = np.repeat(arr, 2048)
  return arr


def hessian_rosen(x):
  return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])


# line-search conditions
def wolfe(f, g, xk, alpha, pk):
  c1 = 1e-4
  return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)


def strong_wolfe(f, g, xk, alpha, pk, c2):
  # typically, c2 = 0.9 when using Newton or quasi-Newton's method.
  #            c2 = 0.1 when using non-linear conjugate gradient method.
  return wolfe(f, g, xk, alpha, pk) and abs(
      np.dot(g(xk + alpha * pk), pk)) <= c2 * abs(np.dot(g(xk), pk))


def gold_stein(f, g, xk, alpha, pk, c):
  return (f(xk) + (1 - c) * alpha * np.dot(g(xk), pk) <= f(xk + alpha * pk)
          ) and (f(xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk))


# line-search step len
def step_length(f, g, xk, alpha, pk, c2):
  return interpolation(f, g,
                       lambda alpha: f(xk + alpha * pk),
                       lambda alpha: np.dot(g(xk + alpha * pk), pk),
                       alpha, c2,
                       lambda f, g, alpha, c2: strong_wolfe(f, g, xk, alpha, pk, c2))


def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha, iters=20):
  # referred implementation here:
  # https://github.com/tamland/non-linear-optimization
  l = 0.0
  h = 1.0
  for i in range(iters):
    if strong_wolfe_alpha(f, g, alpha, c2):
      return alpha

    half = (l + h) / 2
    alpha = - g_alpha(l) * (h**2) / (2 * (f_alpha(h) - f_alpha(l) - g_alpha(l) * h))
    if alpha < l or alpha > h:
      alpha = half
    if g_alpha(alpha) > 0:
      h = alpha
    elif g_alpha(alpha) <= 0:
      l = alpha
  return alpha


# optimization algorithms
def conjugate_gradient(f, g, x0, iterations, error):
  xk = x0
  c2 = 0.1

  fk = f(xk)
  gk = g(xk)
  pk = -gk

  for i in range(iterations):
    alpha = step_length(f, g, xk, 1.0, pk, c2)
    xk1 = xk + alpha * pk
    gk1 = g(xk1)
    beta_k1 = np.dot(gk1, gk1) / np.dot(gk, gk)
    pk1 = -gk1 + beta_k1 * pk

    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break

    xk = xk1
    gk = gk1
    pk = pk1
  return xk, i + 1

if __name__ == '__main__':
  x0 = np.zeros([4096])
  error = 1e-4
  max_iterations = 10000

  #print '\n======= Conjugate Gradient Method ======\n'
  start = time.time()
  x, n_iter = conjugate_gradient(rosenbrock, grad_rosen, x0,
                                 iterations=max_iterations, error=error)
  end = time.time()
