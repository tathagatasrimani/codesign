import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

W = sp.symbols('W')
t = sp.symbols('t')
k = sp.symbols('k')

A = W + t/k
B = t + k*W

u = W/t

alpha = 4
s= 0.6

sigma = 1/(1+sp.exp(-alpha*sp.log(u)))

lam_0 = (1-sigma)*B + sigma*A

delt = (0.5*(k+1/k)-1)*(W*t)**0.5 * sp.exp(-(sp.log(u)**2)/(s**2))

lam = lam_0-delt

k_vals= [0.2, 0.5, 1, 2, 5]

lam_vals = []

for k_i in k_vals:
    lam_vals = []
    W_vals = [10] * 500 + list(np.arange(10.0, 0, -0.02))
    t_vals = list(np.arange(0.02, 10.02, 0.02)) + [10] * 500
    print(len(t_vals))
    for i in range(len(W_vals)):
        sub = {W: W_vals[i], t: t_vals[i], k: k_i}
        lam_vals.append(float(lam.xreplace(sub).evalf()))

    Wlam = [W_vals[i]/lam_vals[i] for i in range(len(W_vals))]
    tlam = [t_vals[i]/lam_vals[i] for i in range(len(t_vals))]

    plt.plot(Wlam, tlam, label=f"k={k_i}")

plt.xlabel("W/lam")
plt.ylabel("t/lam")
plt.legend()
plt.show()

k_vals = [0.2, 0.5, 1, 2, 5]

for k_i in k_vals:
    lam_vals = []
    W_vals = list(np.arange(1000, 0, -1))
    t_vals = list(np.arange(1, 1001, 1.0))
    for i in range(len(W_vals)):
        sub = {W: W_vals[i], t: t_vals[i], k: k_i}
        lam_vals.append(float(lam.xreplace(sub).evalf()))
    plt.plot(t_vals, lam_vals, label=f"k={k_i}")
plt.legend()
plt.xlabel("t")
plt.ylabel("lam")
plt.show()

