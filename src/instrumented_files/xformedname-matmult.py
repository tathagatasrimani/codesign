from loop import loop
import numpy as np


def main():
    a_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    b_1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    c_1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i_1 in range(3):
        for j_1 in range(3):
            loop.start_unroll
            for k_1 in range(3):
                c_1[i_1][j_1] += a_1[i_1][k_1] * b_1[k_1][j_1]
            loop.stop_unroll


if __name__ == '__main__':
    main()
