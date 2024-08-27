from multiprocessing import Pool
from itertools import product
import numpy as np
import pandas as pd

import cacti_util

mem_size_range = [2048, 4096, 32768, 131072, 262144, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 134217728, 67108864, 1073741824]
bw_range = np.concatenate((np.arange(16, 257, 16), np.arange(256, 1025, 64), np.arange(1024, 8193, 256)))

print(f"mem_size_range: {mem_size_range}")
print(f"bw_range: {bw_range}")


def gen_vals_wrapper(data):
    mem_size, bw = data
    try:
        return cacti_util.gen_vals(
            "mem_cache",
            cacheSize=mem_size,
            blockSize=64,
            cache_type="main memory",
            bus_width=bw,
        )
    except Exception as e:
        print(f"Error: {e}")
        return pd.Series()

data = product(mem_size_range, bw_range)
print(list(data))

if __name__ == "__main__":
    with Pool(8) as p:
        res = p.map(
            gen_vals_wrapper,
            zip(
                mem_size_range,
                bw_range,
            ),
        )
    print(res)
    print(f"len(res): {len(res)}")
    print(f"type(res): {type(res)}")
    print(f"len(res[1]): {len(res[1])}")
    print(f"type(res[1]): {type(res[1])}")
    print(f"res[1]: {res[1]}")
    # print(f"res[1].columns: {res[1].columns}")
    print(f"len(res[0]): {len(res[0])}")
    print(f"type(res[0]): {type(res[0])}")
    # print(f"res[0].columns: {res[0].columns}")
    print(f"res[0]: {res[0]}")


# mem_vals = cacti_util.gen_vals(
#     "mem_cache",
#     cacheSize=mem_size_range[0],
#     blockSize=64,
#     cache_type="main memory",
#     bus_width=bw_range[0],
# )
