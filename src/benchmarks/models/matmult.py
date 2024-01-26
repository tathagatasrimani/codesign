from loop import loop
import numpy as np

def main():
    N = 5
    a = np.array(np.random.randint(0, 100, size=(N, N)))
    b = np.array(np.random.randint(0, 100, size=(N, N)))
    d = np.array(np.random.randint(0, 100, size=N))
    c = np.zeros(shape=(N,N)) # output

    # a = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # b = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    # c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # d = [9, 9, 9] # bias vector - testing nvm vs volatile reads.

    # loop().pattern_seek()
    for i in range(N):
        for j in range(N):
            # loop.start_unroll
            for k in range(N):
                # loop().pattern_seek(3)
                c[i][j] += a[i][k] * b[k][j]
            # loop.stop_unroll
            c[i][j] += d[i]

if __name__ == "__main__":
    main()