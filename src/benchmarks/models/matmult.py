from loop import loop
import numpy as np
def main():
    a = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    b = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            loop.start_unroll
            for k in range(3):
                c[i][j] += a[i][k] * b[k][j]
            loop.stop_unroll

if __name__ == "__main__":
    main()