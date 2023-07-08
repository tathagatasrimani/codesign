digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import math
import time
import numpy as np
def rosenbrock(x):...
def grad_rosen(x):...
def hessian_rosen(x):...
def wolfe(f, g, xk, alpha, pk):...
def strong_wolfe(f, g, xk, alpha, pk, c2):...
def gold_stein(f, g, xk, alpha, pk, c):...
def step_length(f, g, xk, alpha, pk, c2):...
def interpolation(f, g, f_alpha, g_alpha, alpha, c2, strong_wolfe_alpha,...
def conjugate_gradient(f, g, x0, iterations, error):...
if __name__ == '__main__':
"]
	56 [label="x0 = np.array([0, 0])
error = 0.0001
max_iterations = 1000
start = time.time()
x, n_iter = conjugate_gradient(rosenbrock, grad_rosen, x0, iterations=
    max_iterations, error=error)
end = time.time()
"]
	"56_calls" [label="np.array
time.time
conjugate_gradient
time.time" shape=box]
	56 -> "56_calls" [label=calls style=dashed]
	1 -> 56 [label="__name__ == '__main__'"]
	subgraph clusterrosenbrock {
		graph [label=rosenbrock]
		3 [label="return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
"]
	}
	subgraph clustergrad_rosen {
		graph [label=grad_rosen]
		7 [label="return np.array([200 * (x[1] - x[0] ** 2) * (-2 * x[0]) + 2 * (x[0] - 1), 
    200 * (x[1] - x[0] ** 2)])
"]
	}
	subgraph clusterhessian_rosen {
		graph [label=hessian_rosen]
		11 [label="return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 *
    x[0], 200]])
"]
	}
	subgraph clusterwolfe {
		graph [label=wolfe]
		15 [label="c1 = 0.0001
return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(g(xk), pk)
"]
	}
	subgraph clusterstrong_wolfe {
		graph [label=strong_wolfe]
		19 [label="return wolfe(f, g, xk, alpha, pk) and abs(np.dot(g(xk + alpha * pk), pk)
    ) <= c2 * abs(np.dot(g(xk), pk))
"]
	}
	subgraph clustergold_stein {
		graph [label=gold_stein]
		23 [label="return f(xk) + (1 - c) * alpha * np.dot(g(xk), pk) <= f(xk + alpha * pk) and f(
    xk + alpha * pk) <= f(xk) + c * alpha * np.dot(g(xk), pk)
"]
	}
	subgraph clusterstep_length {
		graph [label=step_length]
		27 [label="return interpolation(f, g, lambda alpha: f(xk + alpha * pk), lambda alpha:
    np.dot(g(xk + alpha * pk), pk), alpha, c2, lambda f, g, alpha, c2:
    strong_wolfe(f, g, xk, alpha, pk, c2))
"]
	}
	subgraph clusterinterpolation {
		graph [label=interpolation]
		31 [label="l = 0.0
h = 1.0
"]
		32 [label="for i in range(iters):
"]
		33 [label="if strong_wolfe_alpha(f, g, alpha, c2):
"]
		35 [label="return alpha
"]
		33 -> 35 [label="strong_wolfe_alpha(f, g, alpha, c2)"]
		36 [label="half = (l + h) / 2
alpha = -g_alpha(l) * h ** 2 / (2 * (f_alpha(h) - f_alpha(l) - g_alpha(l) * h))
if alpha < l or alpha > h:
"]
		"36_calls" [label="g_alpha
f_alpha
f_alpha
g_alpha" shape=box]
		36 -> "36_calls" [label=calls style=dashed]
		38 [label="alpha = half
"]
		39 [label="if g_alpha(alpha) > 0:
"]
		40 [label="h = alpha
"]
		40 -> 32 [label=""]
		39 -> 40 [label="g_alpha(alpha) > 0"]
		42 [label="if g_alpha(alpha) <= 0:
"]
		43 [label="l = alpha
"]
		43 -> 32 [label=""]
		42 -> 43 [label="g_alpha(alpha) <= 0"]
		42 -> 32 [label="(g_alpha(alpha) > 0)"]
		39 -> 42 [label="(g_alpha(alpha) <= 0)"]
		38 -> 39 [label=""]
		36 -> 38 [label="alpha < l or alpha > h"]
		36 -> 39 [label="(not (alpha < l or alpha > h))"]
		33 -> 36 [label="(not strong_wolfe_alpha(f, g, alpha, c2))"]
		32 -> 33 [label="range(iters)"]
		34 [label="return alpha
"]
		32 -> 34 [label=""]
		31 -> 32 [label=""]
	}
	subgraph clusterconjugate_gradient {
		graph [label=conjugate_gradient]
		48 [label="xk = x0
c2 = 0.1
fk = f(xk)
gk = g(xk)
pk = -gk
"]
		"48_calls" [label="f
g" shape=box]
		48 -> "48_calls" [label=calls style=dashed]
		49 [label="for i in range(iterations):
"]
		50 [label="alpha = step_length(f, g, xk, 1.0, pk, c2)
xk1 = xk + alpha * pk
gk1 = g(xk1)
beta_k1 = np.dot(gk1, gk1) / np.dot(gk, gk)
pk1 = -gk1 + beta_k1 * pk
if np.linalg.norm(xk1 - xk) < error:
"]
		"50_calls" [label="step_length
g
np.dot
np.dot" shape=box]
		50 -> "50_calls" [label=calls style=dashed]
		52 [label="xk = xk1
"]
		51 [label="return xk, i + 1
"]
		52 -> 51 [label=""]
		50 -> 52 [label="np.linalg.norm(xk1 - xk) < error"]
		53 [label="xk = xk1
gk = gk1
pk = pk1
"]
		53 -> 49 [label=""]
		50 -> 53 [label="(np.linalg.norm(xk1 - xk) >= error)"]
		49 -> 50 [label="range(iterations)"]
		49 -> 51 [label=""]
		48 -> 49 [label=""]
	}
}
