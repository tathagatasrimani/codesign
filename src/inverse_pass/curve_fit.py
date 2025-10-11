import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def posynomial_func(x, c1, a1):
    return c1 * (x**a1)

def plot_fit(x_data, y_data, params, filename):
    c1_fit, a1_fit = params
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, posynomial_func(x_data, c1_fit, a1_fit), label='Fitted Curve')
    plt.legend()
    plt.show()
    print(f"showing plot")
    plt.savefig(filename)

def fit_ed_curve(x_data, y_data):
    initial_guesses = [0, -1] # Initial guesses for c1, a1
    params, covariance = curve_fit(posynomial_func, x_data, y_data, p0=initial_guesses)

    # Extract the fitted parameters
    c1_fit, a1_fit = params

    print(f"Fitted c1: {c1_fit}, a1: {a1_fit}")

    return params

if __name__ == "__main__":
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([5, 2.5, 1.25, 0.625, 0.3125]) # Example output from a posynomial
    params = fit_ed_curve(x_data, y_data)
    plot_fit(x_data, y_data, params, 'curve_fit.png')