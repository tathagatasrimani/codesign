from loop import loop
import numpy as np

def main(x, y):
    q = 0.5 + x * y + 1 / 2
    r = x + y
    q = q * r - 3
    w = q + r
    if w < 0:
        a = q + 3
        b = a * r
        r += a + 3 * 2
    else:
        a = q - 3
        b = a / r
    z = [[1, 2, 3, 4, 5, 6]]
    r, q = 2, 3
    for i in range(5):
        # loop.start_unroll
        z[0][i] += z[0][i+1]
    # loop.stop_unroll

def bruh():
    a = 1
    for i in range(3):
        # loop.start_unroll
        a += i
    # loop.stop_unroll

if __name__ == "__main__":
    for i in range(1):
        main(2, 3)
        bruh()
