
import subprocess
import sys
from collections import deque
from copy import deepcopy

from src.cfg_builder import CFGBuilder
from src.hls_instrument import instrument_and_run
import src.hls as hls

BENCHMARK_DIR = "./benchmarks/"
Ghz = 1000*1000*1000

def synthesis_nonai_hardware():
    benchmark = "hpcg"
    loops,sizes = instrument_and_run(BENCHMARK_DIR + "nonai_models/hpcg.py")
    cfg = CFGBuilder().build_from_file("hpcg.py", BENCHMARK_DIR+"nonai_models/hpcg.py",)
    #render CFG
    cfg.build_visual("spmv", "pdf", show=False)
    # modified, the current version doesn't work
    model = hls.map_cfg_to_execution_blocks(cfg, var_sizes=sizes, loop_counts=loops, given_bandwidth=1*Ghz)
    print(model.print_stats())



#synthesis_hardware("aes")
synthesis_nonai_hardware()
