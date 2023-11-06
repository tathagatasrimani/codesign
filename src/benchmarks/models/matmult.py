from loop import loop
import numpy as np
def main():
    a = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    b = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    d = [9, 9, 9] # bias vector - testing nvm vs volatile reads.

    boolean = True

    # loop().pattern_seek()
    for i in range(3):
        for j in range(3):
            loop.start_unroll
            for k in range(3):
                # loop().pattern_seek(3)
                c[i][j] += a[i][k] * b[k][j]

            c[i][j] += d[i]
            loop.stop_unroll

if __name__ == "__main__":
    main()