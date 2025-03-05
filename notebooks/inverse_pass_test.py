import os
import sys
import sympy as sp
import yaml
import concurrent.futures
import time
import logging
logger = logging.getLogger("inverse_test")
os.chdir("..")
sys.path.append(os.getcwd())
from src import sim_util
from src import hw_symbols
from src import optimize

def main():

    logging.basicConfig(filename=f"notebooks/test_files/log.txt", level=logging.INFO)
    start_time = time.time()

    log_dir = "src/tmp"

    