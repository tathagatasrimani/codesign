from loop import loop
import numpy as np


def read_matrices_from_file(N):
    
    a = np.array(np.random.randint(0, 100, size=(N, N)))
    b = np.array(np.random.randint(0, 100, size=(N, N)))
    d = np.array(np.random.randint(0, 100, size=N)) # bias vector - testing nvm vs volatile reads.
    return a, b, d

def main():
    N = 100
    a, b, d = read_matrices_from_file(N)
    c = np.zeros(shape=(N,N)) # output

    for i in range(N):
        for j in range(N):
            loop.start_unroll
            for k in range(N):
                c[i][j] += a[i][k] * b[k][j] + d[j]
            loop.stop_unroll

if __name__ == "__main__":
    main()