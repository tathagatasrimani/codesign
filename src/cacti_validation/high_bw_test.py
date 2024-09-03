# from multiprocessing import Pool
from itertools import product
import numpy as np
import pandas as pd

import cacti_util

if __name__ == "__main__":

    mem_size_range = [2048, 4096, 32768] #, 131072, 262144, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 134217728, 67108864, 1073741824]
    bw_range = [16, 32, 64] #np.concatenate((np.arange(16, 257, 16), np.arange(256, 1025, 64), np.arange(1024, 8193, 256)))

    print(f"mem_size_range: {mem_size_range}")
    print(f"bw_range: {bw_range}")


    data = product(mem_size_range, bw_range)
    # print(list(data))
    # print(list(data))

    res = []
    for mem_size, bw in list(data):
        print(f"calling with mem: {mem_size}, bw: {bw}", flush=True)
        res.append(cacti_util.gen_vals(
            "mem_cache",
            cacheSize=mem_size,
            blockSize=64,
            cache_type="main memory",
            bus_width=bw,
        ))

    # print(res)
