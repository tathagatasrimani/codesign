import argparse
import os
import pprint
import queue
import shutil
import subprocess
import threading
import yaml

from src import codesign
from src import sim_util


class RegressionRun:
    def __init__(self, cfg, codesign_root_dir, single_config_path=None, test_list_path=None, max_parallelism=4):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.single_config_path = single_config_path
        self.test_list_path = test_list_path
        self.max_parallelism = max_parallelism

        self.config_files_to_run = []

        self.base_results_dir = os.path.join(self.codesign_root_dir, "test", "regression_results")

        if self.single_config_path and self.test_list_path:
            raise ValueError("Cannot specify both single_config_path and test_list_path")
        elif self.single_config_path is None and self.test_list_path is None:
            raise ValueError("Must specify either single_config_path or test_list_path")
        elif self.single_config_path is not None:
            
            ## get just the file name from the path and remove the .yaml extension
            single_config_filename = os.path.splitext(os.path.basename(self.single_config_path))[0]

            results_dir = os.path.join(self.base_results_dir, single_config_filename)
            
            self.config_files_to_run.append({"results_dir": results_dir, "config_path": self.single_config_path})
        elif self.test_list_path is not None:
            # load the test list yaml file

            # results dir is base_results_dir/test_list_filename (remove the .yaml)
            self.base_results_dir = os.path.join(self.base_results_dir, os.path.splitext(os.path.basename(self.test_list_path))[0])
            
            with open(self.test_list_path, "r") as f:
                test_list_yaml = yaml.load(f, Loader=yaml.FullLoader)
            base_dir = os.path.dirname(self.test_list_path)
            for test_cfg in test_list_yaml:
                config_path = os.path.join(base_dir, test_cfg)
                results_dir = os.path.join(self.base_results_dir, os.path.splitext(os.path.basename(test_cfg))[0])
                self.config_files_to_run.append({"results_dir": results_dir, "config_path": config_path})

        self.job_queue = queue.Queue()


    def run_regression(self):

        pprint.pprint(self.config_files_to_run)


        # Start worker threads
        threads = []
        for _ in range(self.max_parallelism):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)

        # populate the queue with work
        for config_info in self.config_files_to_run:
            config_path = config_info["config_path"]
            results_dir = config_info["results_dir"]
            print(f"Running config file: {config_path} with results dir: {results_dir}")
            self.run_single_config_file(config_path, results_dir)

        # Wait for all jobs to finish
        self.job_queue.join()

        # Stop workers
        for _ in range(self.max_parallelism):
            self.job_queue.put(None)
        for t in threads:
            t.join()
        print("All jobs complete.")
        
    
    def run_single_config_file(self, config_file_path, results_dir):
        ## read through the config file and run each job specified in it
        with open(config_file_path, "r") as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        
        for config_name in config_yaml:
            print(f"Running config: {config_name}")
            job_results_dir = os.path.join(results_dir, config_name)
            self.job_queue.put((config_file_path, config_name, job_results_dir))
    
    
    def run_single_job(self, config_path, config_name, job_results_dir):
        # create results dir if it doesn't exist or clear it if it does
        if os.path.exists(job_results_dir):
            shutil.rmtree(job_results_dir)
        
        os.makedirs(job_results_dir)

        # copy the config file to the results dir
        shutil.copy(config_path, os.path.join(job_results_dir, "config.yaml"))
        
        log_path = os.path.join(job_results_dir, "run_codesign.log")
        
        # start a subprocess that runs codesign with the config file
        cmd = (
            f'bash -c "shopt -s expand_aliases && '
            f'source full_env_start.sh && '
            f'$(which python) -u -m src.codesign '
            f'--config {config_name} '
            f'--additional_cfg_file {os.path.join(job_results_dir, "config.yaml")} '
            f'--tmp_dir {os.path.join(job_results_dir, "tmp")} '
            f'-f {os.path.join(job_results_dir, "log")} "'
        )
        # Run in shell mode so it inherits environment vars (PATH, conda env, etc.)

        # capture stdout/stderr to a logfile in the results dir
        
        with open(log_path, "wb") as logf:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.codesign_root_dir,   # optional: run inside the results dir
                env=os.environ,    # copy parent environment
                stdout=logf,
                stderr=subprocess.STDOUT,
                executable="/bin/bash",
            )
            # wait for the process to complete
            process.wait()

        print(f"codesign output captured to: {log_path}")
 
        if process.returncode != 0:
            raise RuntimeError(f"codesign failed with exit code {process.returncode}")

        print("Codesign run complete!")

    def worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            config_path, config_name, results_dir = job
            try:
                self.run_single_job(config_path, config_name, results_dir)
            except Exception as e:
                print(f"❌ Job {config_name} failed: {e}")
            finally:
                self.job_queue.task_done()

        


if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Codesign Regression Run",
            description="Runs the codesign framework in regression mode.",
            epilog="the end",
        )
        parser.add_argument(
            "-s",
            "--single_config",
            type=str,
            help="Run a single config. Specify the path."
        )

        parser.add_argument(
            "-l",
            "--test_list",
            type=str,
            help="Runs a testlist yaml file specifying multiple configs to run. The configs must be in the same directory as the testlist yaml file. Specify the path."
        )

        parser.add_argument(
            "-m",
            "--max_parallelism",
            type=int,
            help="The maximum number of tests that should run in parallel.",
            default=4,
        )

        args = parser.parse_args()

        ## check if we are in the codesign directory
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(cwd, "src")) or not os.path.exists(os.path.join(cwd, "test")):
            print("Error: must be run from the codesign root directory")
            exit(1)

        reg_run = RegressionRun(cfg=None, codesign_root_dir=cwd, single_config_path=args.single_config, test_list_path=args.test_list, max_parallelism=args.max_parallelism)

        reg_run.run_regression()

    

