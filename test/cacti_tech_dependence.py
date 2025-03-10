from typing import List
from argparse import ArgumentParser
from glob import glob
import os
import sympy as sp
import multiprocessing as mp

from src import cacti_util, CACTI_DIR
from src.cacti import TRANSISTOR_SIZES


def test_one_point(cfg, transistor_size):
    print(f"Testing {cfg} at {transistor_size} nm")

    buf_vals = cacti_util.gen_vals(cfg, transistor_size=transistor_size)
    # print(f"buf_vals: {buf_vals}")
    buf_opt = {
        "ndwl": buf_vals["Ndwl"],
        "ndbl": buf_vals["Ndbl"],
        "nspd": buf_vals["Nspd"],
        "ndcm": buf_vals["Ndcm"],
        "ndsam1": buf_vals["Ndsam_level_1"],
        "ndsam2": buf_vals["Ndsam_level_2"],
        "repeater_spacing": buf_vals["Repeater spacing"],
        "repeater_size": buf_vals["Repeater size"],
    }
    IO_info = cacti_util.gen_symbolic(f"{args.config}_{int(transistor_size*1e3)}", f"cfg/{cfg}.cfg", buf_opt, use_piecewise=False)


def compare(file1, file2, queue: mp.Queue):
    expr1 = sp.sympify(open(file1).read())
    expr2 = sp.sympify(open(file2).read())
    queue.put((file1, file2, sp.simplify(expr1 - expr2) == 0))
    return
    # return sp.simplify(expr1 - expr2) == 0

def test_all_points(cfgs: List, transistor_sizes: List):
    for cfg in cfgs:
        print(f"Testing {cfg}")
        Q = mp.Queue()
        files = []
        for ts in transistor_sizes:
            test_one_point(cfg, ts)
            files.append(glob(os.path.join(CACTI_DIR, "symbolic_expressions", f"{args.config}_{int(ts*1e3)}*")))
        # print(f"files: {files}")
        files = [f for sublist in files for f in sublist]

        at_files = [f for f in files if "access_time" in f]
        rd_files = [f for f in files if "read_dynamic" in f]
        wd_files = [f for f in files if "write_dynamic" in f]
        l_files = [f for f in files if "leakage" in f]

        at_processes = []
        # print(f"at_files: {at_files}")
        # expressions = [sp.sympify(open(f).read()) for f in at_files]
        for i in range(len(at_files)):
            for j in range(i+1, len(at_files)):
                print(f"Comparing {at_files[i].split('/')[-1]} and {at_files[j].split('/')[-1]}")
                at_processes.append(mp.Process(target=compare, args=(at_files[i], at_files[j], Q)))
                at_processes[-1].start()
                # print(f"Expressions equal: {sp.simplify(expressions[i] - expressions[j]) == 0}")
        # print(f"rd_files: {rd_files}")
        # expressions = [sp.sympify(open(f).read()) for f in rd_files]
        rd_processes = []
        for i in range(len(rd_files)):
            for j in range(i+1, len(rd_files)):
                print(f"Comparing {rd_files[i].split('/')[-1]} and {rd_files[j].split('/')[-1]}")
                rd_processes.append(mp.Process(target=compare, args=(rd_files[i], rd_files[j], Q)))
                rd_processes[-1].start()
                # print(f"Expressions equal: {sp.simplify(expressions[i] - expressions[j]) == 0}")
        # print(f"wd_files: {wd_files}")
        # expressions = [sp.sympify(open(f).read()) for f in wd_files]
        wd_processes = []
        for i in range(len(wd_files)):
            for j in range(i+1, len(wd_files)):
                print(f"Comparing {wd_files[i].split('/')[-1]} and {wd_files[j].split('/')[-1]}")
                wd_processes.append(mp.Process(target=compare, args=(wd_files[i], wd_files[j], Q)))
                wd_processes[-1].start()
                # print(f"Expressions equal: {sp.simplify(expressions[i] - expressions[j]) == 0}")
        # print(f"l_files: {l_files}")
        # expressions = [sp.sympify(open(f).read()) for f in l_files]
        l_processes = []
        for i in range(len(l_files)):
            for j in range(i+1, len(l_files)):
                print(f"Comparing {l_files[i].split('/')[-1]} and {l_files[j].split('/')[-1]}")
                l_processes.append(mp.Process(target=compare, args=(l_files[i], l_files[j], Q)))
                l_processes[-1].start()
                # print(f"Expressions equal: {sp.simplify(expressions[i] - expressions[j]) == 0}")
        
        for p in at_processes:
            p.join()
        for p in rd_processes:
            p.join()
        for p in wd_processes:
            p.join()
        for p in l_processes:
            p.join()
        print("All processes joined")
        print("Results:")
        while not Q.empty():
            print(Q.get())
        Q.close()
    

        

if __name__ == "__main__":
    parser = ArgumentParser(description="Specify config (--config) and data (--dat) to test; both optional, will do all if not provided.")
    parser.add_argument("-c", "--config", type=str, default="base_cache", help="Path or Name to the configuration file; don't prepend src/cacti/ or .cfg")
    # parser.add_argument("-t", "--transistor_size", type=int,  help="Specify technology in nm -> e.g. 90; if not provdied, do 22, 32, 45, 65, 90, and 180")
    args = parser.parse_args()

    mp.set_start_method('spawn')

    if args.config:
        cfgs = [args.config]
    else:
        cfgs = glob(os.path.join(CACTI_DIR, "cfg", "*.cfg"))
    print(cfgs)

    # if args.dat:
    #     transistor_sizes = [args.dat * 1e-3]
    # else:
    transistor_sizes = TRANSISTOR_SIZES[::2]
    print(transistor_sizes)

    test_all_points(cfgs, transistor_sizes)
    # ["base_cache", "base_ram", "base_cam", "base_fifo", "base_register_file", "base_sram"]
