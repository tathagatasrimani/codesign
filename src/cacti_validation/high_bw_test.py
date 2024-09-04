from multiprocessing import Pool
from itertools import product
import numpy as np
import pandas as pd

import cacti_util

def gen_vals_wrapper(mem_size, bw):
    # mem_size = data
    # bw = 16
    print(f"running mem size: {mem_size} with bw: {bw}", flush=True)
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

def square(x):
    return x**2

# if __name__ == "__main__":


#     with Pool(8) as p:
#         res = p.map(square, [1, 2, 3])
#         print(res)
#         p.close()
#         p.join()

if __name__ == "__main__":

    mem_size_range = [2048, 4096, 32768, 131072]#, 262144, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 134217728, 67108864, 1073741824]
    bw_range = [16, 32, 64] #np.concatenate((np.arange(16, 257, 16), np.arange(256, 1025, 64), np.arange(1024, 8193, 256)))

    print(f"mem_size_range: {mem_size_range}")
    print(f"bw_range: {bw_range}")


    data = product(mem_size_range, bw_range)
    # print(list(data))

    with Pool(1) as p:
        res = p.starmap(
            gen_vals_wrapper,
            data,
        )
        print(f"res: {res}")
        # print(f"res.ready: {res.ready()}")
        # res = res.get()
        # print(f"res.succesful: {res.successful()}")

        p.close()
        p.join()
        print(f"res: {res}")
    print(res)
    for param, series in zip(data, res):
        print(series)
        series['mem size'] = param[0]
        series['bw'] = param[1]
    res_df = pd.concat(res)
    res_df.to_csv("cacti_validation/res.csv")
    # with open("cacti_validation/res.txt", "w") as f:
    #     pd.concat()
    #     [f.write(str(series)) for series in res]
    # print(f"len(res): {len(res)}")
    # print(f"type(res): {type(res)}")
    # print(f"len(res[1]): {len(res[1])}")
    # print(f"type(res[1]): {type(res[1])}")
    # print(f"res[1]: {res[1]}")
    # # print(f"res[1].columns: {res[1].columns}")
    # print(f"len(res[0]): {len(res[0])}")
    # print(f"type(res[0]): {type(res[0])}")
    # # print(f"res[0].columns: {res[0].columns}")
    # print(f"res[0]: {res[0]}")

    # print(res)
