import argparse
import os
import pprint
import queue
import shutil
import subprocess
import sys
import threading
import time as _time
import yaml
import signal
import re

from src import codesign
from src import sim_util

MAX_JOB_RUNTIME_MINS = 70  # in minutes

CHECKPOINT_RE = re.compile(
    r"CHECKPOINT\s+REACHED:\s*([A-Za-z0-9_\-\/]+)\s*FOR\s+ITERATION:\s*(\d+)"
    r"(?:\s*at\s*TIME:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}))?"
)

# constants for initialization phase. This is the first phase/iteration count for a run. 
INITIAL_PHASE_CHECKPOINTS = {"setup"}
INITIAL_PHASE_ITERATIONS = {0}



class RegressionRun:
    def __init__(self, cfg, codesign_root_dir, single_config_path=None, test_list_path=None, max_parallelism=4, absolute_paths=False):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.single_config_path = single_config_path
        self.test_list_path = test_list_path
        self.max_parallelism = max_parallelism
        self.absolute_paths = absolute_paths

        self.completed_jobs = 0
        self.passed_jobs = 0
        self.failed_jobs = 0
        self.completed_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.job_worker_lock = threading.Lock()
        self.job_start_times = {} # job_name -> start_time
        self.job_finish_times = {} # job_name -> finish_time
        self.job_worker_map = {} # which worker is working on which job
        self.job_checkpoints = {}  # job_name -> {"latest": (step:str, iter:int), "history": [(step, iter, timestamp)]}
        self.job_initialized = {}  # job_name -> bool (True once setup@0 seen)

    
        self.config_files_to_run = []

        self.base_results_dir = os.path.join(self.codesign_root_dir, "test", "regression_results")

        self.results_file = os.path.join(self.base_results_dir, "regression_results.yaml")
        self.results_data = {}
        self.results_lock = threading.Lock()

        if self.single_config_path and self.test_list_path:
            raise ValueError("Cannot specify both single_config_path and test_list_path")
        elif self.single_config_path is None and self.test_list_path is None:
            raise ValueError("Must specify either single_config_path or test_list_path")
        elif self.single_config_path is not None:
            if not self.absolute_paths:
                # make path relative to codesign_root_dir/test/regressions/
                self.single_config_path = os.path.join(self.codesign_root_dir, "test", "regressions", self.single_config_path)
            ## get just the file name from the path and remove the .yaml extension
            single_config_filename = os.path.splitext(os.path.basename(self.single_config_path))[0]

            results_dir = os.path.join(self.base_results_dir, single_config_filename)
            
            self.config_files_to_run.append({"results_dir": results_dir, "config_path": self.single_config_path})
        elif self.test_list_path is not None:
            # load the test list yaml file

            if not self.absolute_paths:
                # make path relative to codesign_root_dir/test/regressions/
                self.test_list_path = os.path.join(self.codesign_root_dir, "test", "regressions", self.test_list_path)

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

        self.regression_start_ts = _time.time()

        # Start worker threads
        threads = []
        for worker_id in range(self.max_parallelism):
            t = threading.Thread(target=self.worker, args=(worker_id,))
            t.start()
            threads.append(t)

        # populate the queue with work
        total_jobs = 0
        for config_info in self.config_files_to_run:
            config_path = config_info["config_path"]
            results_dir = config_info["results_dir"]
            total_jobs += self.run_single_config_file(config_path, results_dir)

        print(f"Queued {self.job_queue.qsize()} jobs across {self.max_parallelism} workers.")

        # Start progress tracker
        tracker_thread = threading.Thread(target=self.progress_tracker, args=(total_jobs,), daemon=True)
        tracker_thread.start()
        
        # Wait for all jobs to finish
        self.job_queue.join()

        # Stop workers
        self.stop_event.set()
        for _ in range(self.max_parallelism):
            self.job_queue.put(None)
        for t in threads:
            t.join()
        tracker_thread.join()
        
    
    def run_single_config_file(self, config_file_path, results_dir):
        ## read through the config file and run each job specified in it
        with open(config_file_path, "r") as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        
        count = 0
        for config_name in config_yaml:
            job_results_dir = os.path.join(results_dir, config_name)
            self.job_queue.put((config_file_path, config_name, job_results_dir))
            count+=1

        return count
    
    
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
        # capture stdout/stderr to a logfile in the results dir and enforce a timeout
        timeout_secs = MAX_JOB_RUNTIME_MINS * 60
        with open(log_path, "wb") as logf:
            # start process in its own process group so we can kill children reliably
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.codesign_root_dir,
                env=os.environ,
                stdout=logf,
                stderr=subprocess.STDOUT,
                executable="/bin/bash",
                preexec_fn=os.setsid,
            )
            try:
                process.wait(timeout=timeout_secs)
            except subprocess.TimeoutExpired:
                # log and terminate the whole process group
                msg = f"\n*** Job {config_name} exceeded max runtime ({MAX_JOB_RUNTIME_MINS} mins) and was terminated ***\n"
                try:
                    logf.write(msg.encode())
                    logf.flush()
                except Exception:
                    pass
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
                # give it a short grace period to exit
                try:
                    process.wait(timeout=10)
                except Exception:
                    pass
                success = False
            else:
                success = self.check_correct_completion(log_path)
 
        with self.completed_lock:
            self.completed_jobs += 1

        return success

    def worker(self, worker_id):
        while True:
            job = self.job_queue.get()
            if job is None:
                break

            config_path, config_name, results_dir = job
            start_time = _time.time()
            with self.job_worker_lock:
                self.job_start_times[config_name] = start_time
                self.job_worker_map[config_name] = worker_id

            # launch the job and a log watcher
            log_path = os.path.join(results_dir, "run_codesign.log")
            stop_evt = threading.Event()
            watcher = threading.Thread(
                target=self._watch_log_for_checkpoints,
                args=(config_name, log_path, stop_evt),
                daemon=True,
            )
            watcher.start()

            try:
                success = self.run_single_job(config_path, config_name, results_dir)
            except Exception:
                success = False
            finally:
                finish_time = _time.time()
                with self.job_worker_lock:
                    self.job_finish_times[config_name] = finish_time
                elapsed = finish_time - start_time

                # update pass/fail counters here (since run_single_job returns success)
                with self.completed_lock:
                    if success:
                        self.passed_jobs += 1
                    else:
                        self.failed_jobs += 1

                # stop watcher & write YAML result
                stop_evt.set()
                watcher.join(timeout=2.0)

                self.record_result(config_name, success, elapsed)
                self.job_queue.task_done()




    def check_correct_completion(self, log_path):

        # look for "^^CHECKPOINT REACHED: <checkpoint_step> . RUN SUCCEEDED^^ 
        success = False
        with open(log_path, "r") as f:
            for line in f:
                if "^^CHECKPOINT REACHED:" in line and ". RUN SUCCEEDED^^" in line:
                    success = True
                    break

        return success


    # ----------------------------
    # Progress animation thread
    # ----------------------------
    def progress_tracker(self, total_jobs):
        spinner = ['|', '/', '-', '\\']
        i = 0
        sys.stdout.write("\n" * (self.max_parallelism + 2))
        sys.stdout.flush()

        while not self.stop_event.is_set():
            with self.completed_lock:
                done = self.completed_jobs
                passed = self.passed_jobs
                failed = self.failed_jobs

            percent_done = (done / total_jobs) * 100 if total_jobs > 0 else 0
            passed_pct = (passed / total_jobs) * 100 if total_jobs > 0 else 0
            failed_pct = (failed / total_jobs) * 100 if total_jobs > 0 else 0

            bar_len = 40
            filled = int(bar_len * done / total_jobs)
            bar = "█" * filled + "-" * (bar_len - filled)

            sys.stdout.write(f"\033[{self.max_parallelism + 2}F")
            now = _time.time()

            with self.job_worker_lock:
                # build a reverse map: worker_id -> running job (exclude finished jobs)
                running_by_worker = {wid: None for wid in range(self.max_parallelism)}
                for job_name, wid in self.job_worker_map.items():
                    if job_name not in self.job_finish_times:
                        running_by_worker[wid] = job_name

            now = _time.time()
            for wid in range(self.max_parallelism):
                job_name = running_by_worker[wid]
                if job_name:
                    start_time = self.job_start_times.get(job_name)
                    if start_time:
                        elapsed = int(now - start_time)
                        mins, secs = divmod(elapsed, 60)
                    else:
                        mins, secs = 0, 0
                    ckpt_txt = ""
                    with self.job_worker_lock:
                        latest = self.job_checkpoints.get(job_name, {}).get("latest")
                    if latest:
                        step, it = latest
                        initialized = self.job_initialized.get(job_name, False)
                        if not initialized:
                            ckpt_txt = " | Initializing..."
                        else:
                            ckpt_txt = f" | Checkpoint {step}@{it}"
                    else:
                        ckpt_txt = " | Initializing..."
                    sys.stdout.write(f"\033[KWorker {wid+1}: Running {job_name} | Time {mins:02d}:{secs:02d}{ckpt_txt}\n")
                else:
                    sys.stdout.write(f"\033[KWorker {wid+1}: Idle\n")

            sys.stdout.write(
                f"\033[K{spinner[i % len(spinner)]}  [{bar}] {done}/{total_jobs} jobs complete "
                f"({percent_done:.1f}%) | Passed ✅: {passed} ({passed_pct:.1f}%) | "
                f"Failed ❌: {failed} ({failed_pct:.1f}%)\n\n"
            )
            sys.stdout.flush()
            i += 1
            _time.sleep(0.2)

        sys.stdout.write(f"\033[{self.max_parallelism + 2}F")
        for wid in range(self.max_parallelism):
            sys.stdout.write(f"\033[KWorker {wid+1}: Idle\n")
        sys.stdout.write(
            f"\033[K✅  [████████████████████████████████████████] All jobs complete!  "
            f"Passed ✅: {self.passed_jobs} ({(self.passed_jobs/total_jobs)*100:.1f}%) | "
            f"Failed ❌: {self.failed_jobs} ({(self.failed_jobs/total_jobs)*100:.1f}%)\n\n"
        )

    def record_result(self, job_name, success, elapsed_time):
        """Record job result and write to regression_results.yaml safely."""
        with self.results_lock:
            self.results_data[job_name] = {
                "status": "passed" if success else "failed",
                "elapsed_time_sec": round(elapsed_time, 2),
            }
            # ensure base dir exists
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            # atomic write to avoid corruption
            tmp_path = self.results_file + ".tmp"
            with open(tmp_path, "w") as f:
                yaml.dump(self.results_data, f, default_flow_style=False)
            os.replace(tmp_path, self.results_file)

    def _watch_log_for_checkpoints(self, job_name, log_path, stop_evt):
        last_size = 0
        last_mtime = 0

        while not stop_evt.is_set():
            if not os.path.exists(log_path):
                _time.sleep(0.5)
                continue

            try:
                size = os.path.getsize(log_path)
                mtime = os.path.getmtime(log_path)
                if size < last_size or mtime != last_mtime:
                    last_size = 0

                with open(log_path, "r", errors="ignore") as f:
                    f.seek(last_size)
                    for line in f:
                        m = CHECKPOINT_RE.search(line)
                        if not m:
                            continue

                        step = m.group(1)
                        it = int(m.group(2))
                        # Parse optional timestamp on the line
                        ts_str = m.group(3)
                        parsed_ts = None
                        if ts_str:
                            try:
                                # ts_str like "2025-11-07 14:38:08" (local time)
                                from datetime import datetime
                                parsed_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                                # convert to epoch seconds in local time
                                import time as __t
                                parsed_ts = __t.mktime(parsed_dt.timetuple())
                            except Exception:
                                parsed_ts = None

                        now = _time.time()
                        with self.job_worker_lock:
                            data = self.job_checkpoints.setdefault(job_name, {"latest": None, "history": []})
                            data["latest"] = (step, it)
                            data["history"].append((step, it, parsed_ts if parsed_ts is not None else now))

                            # Only mark initialized if we saw 'setup @ 0' WITH a timestamp
                            # that is on/after the regression start.
                            if step.lower() == "setup" and it == 0 and parsed_ts is not None:
                                if parsed_ts >= getattr(self, "regression_start_ts", 0):
                                    self.job_initialized[job_name] = True

                    last_size = f.tell()
                    last_mtime = mtime
            except Exception:
                pass

            _time.sleep(0.5)



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
            "-a",
            "--absolute_paths",
            action="store_true",
            help="Path arguments provided on command line are relative to the codesign root directory instead of ~/test/regressions/",
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

        reg_run = RegressionRun(cfg=None, codesign_root_dir=cwd, single_config_path=args.single_config, test_list_path=args.test_list, max_parallelism=args.max_parallelism, absolute_paths=args.absolute_paths)

        reg_run.run_regression()



