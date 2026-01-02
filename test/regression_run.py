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

MAX_JOB_RUNTIME_MINS = 180  # in minutes

NO_BETTER_DESIGN_POINT_FOUND_MSG = "FLOW END: No better design point found."

CHECKPOINT_RE = re.compile(
    r"CHECKPOINT\s+REACHED:\s*([A-Za-z0-9_\-\/]+)\s*FOR\s+ITERATION:\s*(\d+)"
    r"(?:\s*at\s*TIME:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}))?"
)

# constants for initialization phase. This is the first phase/iteration count for a run. 
INITIAL_PHASE_CHECKPOINTS = {"setup"}
INITIAL_PHASE_ITERATIONS = {0}

# =====================================================================
#               CTRL-C CLEAN SHUTDOWN SUPPORT
# =====================================================================

ACTIVE_PROCESSES = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()
regression_instance = None

def register_process(p):
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.add(p)

def unregister_process(p):
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.discard(p)

def kill_all_processes():
    with ACTIVE_PROCESSES_LOCK:
        for p in list(ACTIVE_PROCESSES):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass

def sigint_handler(signum, frame):
    print("\n[CTRL-C] Stopping all jobs...")
    sys.stdout.flush()
    kill_all_processes()

    global regression_instance
    if regression_instance:
        regression_instance.stop_event.set()

        # unstick blocked workers
        for _ in range(regression_instance.max_parallelism):
            regression_instance.job_queue.put(None)

        while not regression_instance.job_queue.empty():
            try:
                regression_instance.job_queue.get_nowait()
                regression_instance.job_queue.task_done()
            except queue.Empty:
                break

signal.signal(signal.SIGINT, sigint_handler)

# =====================================================================



class RegressionRun:
    def __init__(self, cfg, codesign_root_dir, single_config_path=None, test_list_path=None, max_parallelism=4, absolute_paths=False, silent_mode=False, github_autotest_mode=False, preinstalled_openroad_path=None, dsp_sweep=None):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.single_config_path = single_config_path
        self.test_list_path = test_list_path
        self.max_parallelism = max_parallelism
        self.absolute_paths = absolute_paths
        self.silent_mode = silent_mode
        self.preinstalled_openroad_path = preinstalled_openroad_path
        self.github_autotest_mode = github_autotest_mode
        # Format: "start:end:step" e.g., "10:2000:10"
        self.dsp_sweep = dsp_sweep  

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

            self.results_file = os.path.join(results_dir, "regression_results_full.yaml")
            self.results_file_summary = os.path.join(results_dir, "regression_results_summary.yaml")
            
            self.config_files_to_run.append({"results_dir": results_dir, "config_path": self.single_config_path})
        elif self.test_list_path is not None:
            # load the test list yaml file

            if not self.absolute_paths:
                # make path relative to codesign_root_dir/test/regressions/
                self.test_list_path = os.path.join(self.codesign_root_dir, "test", "regressions", self.test_list_path)

            # results dir is base_results_dir/test_list_filename (remove the .yaml)
            self.base_results_dir = os.path.join(self.base_results_dir, os.path.splitext(os.path.basename(self.test_list_path))[0])

            self.results_file = os.path.join(self.base_results_dir, "regression_results_full.yaml")
            self.results_file_summary = os.path.join(self.base_results_dir, "regression_results_summary.yaml")
            
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
            if self.stop_event.is_set():
                break 
            config_path = config_info["config_path"]
            results_dir = config_info["results_dir"]
            total_jobs += self.run_single_config_file(config_path, results_dir)

        # print(f"Queued {self.job_queue.qsize()} jobs across {self.max_parallelism} workers.")

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

        self.write_summary_results()

        ## return 0 if all tests passed, 1 if any test failed
        return 1 if self.failed_jobs > 0 else 0
        
    
    def run_single_config_file(self, config_file_path, results_dir):
        ## read through the config file and run each job specified in it
        with open(config_file_path, "r") as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        
        count = 0
        for config_name in config_yaml:
            if self.stop_event.is_set():
                break
            
            # Check if DSP sweep is enabled
            if self.dsp_sweep:
                # Parse sweep parameters
                try:
                    start, end, step = map(int, self.dsp_sweep.split(':'))
                except ValueError:
                    raise ValueError("DSP sweep must be in format 'start:end:step', e.g., '10:2000:10'")
                
                # Generate jobs for each DSP value in the sweep
                for dsp_val in range(start, end + 1, step):
                    sweep_config_name = f"{config_name}_dsp{dsp_val}"
                    job_results_dir = os.path.join(results_dir, sweep_config_name)
                    self.job_queue.put((config_file_path, config_name, job_results_dir, dsp_val))
                    count += 1
            else:
                # Original behavior - no sweep
                job_results_dir = os.path.join(results_dir, config_name)
                self.job_queue.put((config_file_path, config_name, job_results_dir, None))
                count += 1

        return count
    
    
    def run_single_job(self, config_path, config_name, job_results_dir, dsp_override=None):
        # create results dir if it doesn't exist or clear it if it does
        if os.path.exists(job_results_dir):
            shutil.rmtree(job_results_dir)
        
        os.makedirs(job_results_dir)

        # copy the config file to the results dir
        shutil.copy(config_path, os.path.join(job_results_dir, "config.yaml"))
        
        # If DSP override is provided, modify the config
        if dsp_override is not None:
            config_copy_path = os.path.join(job_results_dir, "config.yaml")
            with open(config_copy_path, "r") as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            
            # Override max_dsp in the specific config
            if config_name in config_data and 'args' in config_data[config_name]:
                config_data[config_name]['args']['max_dsp'] = dsp_override
            
            with open(config_copy_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
        
        log_path = os.path.join(job_results_dir, "run_codesign.log")


        if self.preinstalled_openroad_path is not None:
            openroad_preinstalled_flag = f'OPENROAD_PRE_INSTALLED=1 && '
            openroad_preinstalled_path_option = f'--preinstalled_openroad_path {self.preinstalled_openroad_path} '
        else:
            openroad_preinstalled_flag = ''
            openroad_preinstalled_path_option = ''

        
        # start a subprocess that runs codesign with the config file
        cmd = (
            f'bash -c "shopt -s expand_aliases && '
            f'{openroad_preinstalled_flag}'
            f'source full_env_start.sh && '
            f'$(which python) -u -m src.codesign '
            f'--config {config_name} '
            f'--additional_cfg_file {os.path.join(job_results_dir, "config.yaml")} '
            f'--tmp_dir {os.path.join(job_results_dir, "tmp")} '
            f'{openroad_preinstalled_path_option}'
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
            
            register_process(process)
            
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
            finally:
                unregister_process(process)
 
        with self.completed_lock:
            self.completed_jobs += 1

        return success

    def worker(self, worker_id):
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.stop_event.is_set():
                self.job_queue.task_done()
                break

            if job is None:
                self.job_queue.task_done()
                break

            config_path, config_name, results_dir, dsp_override = job
            
            # Use modified config name if DSP sweep is active
            display_name = config_name if dsp_override is None else f"{config_name}_dsp{dsp_override}"
            
            start_time = _time.time()
            with self.job_worker_lock:
                self.job_start_times[display_name] = start_time
                self.job_worker_map[display_name] = worker_id

            log_path = os.path.join(results_dir, "run_codesign.log")
            stop_evt = threading.Event()
            watcher = threading.Thread(
                target=self._watch_log_for_checkpoints,
                args=(display_name, log_path, stop_evt),
                daemon=True,
            )
            watcher.start()

            try:
                success = self.run_single_job(config_path, config_name, results_dir, dsp_override)
            except Exception:
                success = False
            finally:
                finish_time = _time.time()
                with self.job_worker_lock:
                    self.job_finish_times[display_name] = finish_time
                elapsed = finish_time - start_time

                with self.completed_lock:
                    if success:
                        self.passed_jobs += 1
                    else:
                        self.failed_jobs += 1

                stop_evt.set()
                watcher.join(timeout=2.0)

                self.record_result(display_name, success, elapsed)
                self.job_queue.task_done()

            if self.stop_event.is_set():
                break




    def check_correct_completion(self, log_path):
        ''' check the log file to see if the program completed successfully '''
        # look for "^^CHECKPOINT REACHED: <checkpoint_step> . RUN SUCCEEDED^^ 
        success = False
        with open(log_path, "r") as f:
            for line in f:
                if "^^CHECKPOINT REACHED:" in line and ". RUN SUCCEEDED^^" in line or NO_BETTER_DESIGN_POINT_FOUND_MSG in line:
                    success = True
                    break

        return success


    # ----------------------------
    # Progress animation thread
    # ----------------------------
    def progress_tracker(self, total_jobs):
        spinner = ['|', '/', '-', '\\']
        i = 0
        last_update_time = 0

        github_mode = self.github_autotest_mode
        update_interval = 30.0 if github_mode else 0.2  # 30s snapshot interval for GitHub

        # Build a clean header message
        if self.test_list_path:
            header_target = self.test_list_path
        elif self.single_config_path:
            header_target = self.single_config_path
        else:
            header_target = "(unknown test source)"

        startup_msg = f"Running regression test: {header_target}\nUsing {self.max_parallelism} worker threads.\n"

        if github_mode:
            print("Running in GitHub autotest mode — progress updates every 30 seconds.\n")
            print(startup_msg)
            sys.stdout.flush()
            _time.sleep(5.0)  # delay before first GitHub progress print

        elif not self.silent_mode:
            # Clear terminal for clean live progress display
            os.system('clear' if os.name != 'nt' else 'cls')

            # Print the header *before* reserving lines for the tracker
            sys.stdout.write(startup_msg + "\n")
            # Reserve lines for tracker display region below the header
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

            now = _time.time()
            if not github_mode and not self.silent_mode:
                sys.stdout.write(f"\033[{self.max_parallelism + 2}F")

            # Gather running jobs per worker
            with self.job_worker_lock:
                running_by_worker = {wid: None for wid in range(self.max_parallelism)}
                for job_name, wid in self.job_worker_map.items():
                    if job_name not in self.job_finish_times:
                        running_by_worker[wid] = job_name

            # -----------------------------
            # Local interactive terminal mode
            # -----------------------------
            if not github_mode and not self.silent_mode:
                for wid in range(self.max_parallelism):
                    job_name = running_by_worker[wid]
                    if job_name:
                        start_time = self.job_start_times.get(job_name)
                        elapsed = int(now - start_time) if start_time else 0
                        mins, secs = divmod(elapsed, 60)
                        ckpt_txt = ""
                        latest = self.job_checkpoints.get(job_name, {}).get("latest")
                        if latest:
                            step, it = latest
                            if not self.job_initialized.get(job_name, False):
                                ckpt_txt = " | Initializing..."
                            else:
                                ckpt_txt = f" | {step} completed for iteration {it}"
                        else:
                            ckpt_txt = " | Initializing..."
                        sys.stdout.write(f"\033[KWorker {wid+1}: Running {job_name} | Time {mins:02d}:{secs:02d}{ckpt_txt}\n")
                    else:
                        sys.stdout.write(f"\033[KWorker {wid+1}: Idle\n")

                sys.stdout.write(
                    f"\033[K{spinner[i % len(spinner)]}  [{bar}] {done}/{total_jobs} complete "
                    f"({percent_done:.1f}%) | ✅ {passed} ({passed_pct:.1f}%) | ❌ {failed} ({failed_pct:.1f}%)\n\n"
                )
                sys.stdout.flush()

            # -----------------------------
            # GitHub mode — periodic snapshot
            # -----------------------------
            elif github_mode and (now - last_update_time >= update_interval):
                print("------------------------------------------------------------")
                print(f"[Progress] {done}/{total_jobs} complete ({percent_done:.1f}%)")
                print(f"✅ Passed: {passed} ({passed_pct:.1f}%) | ❌ Failed: {failed} ({failed_pct:.1f}%)")
                print(f"[{bar}]")
                print("Worker Status:")
                for wid in range(self.max_parallelism):
                    job_name = running_by_worker[wid]
                    if job_name:
                        start_time = self.job_start_times.get(job_name)
                        elapsed = int(now - start_time) if start_time else 0
                        mins, secs = divmod(elapsed, 60)
                        latest = self.job_checkpoints.get(job_name, {}).get("latest")
                        if latest:
                            step, it = latest
                            ckpt_txt = f" | {step} completed for iteration {it}"
                        else:
                            ckpt_txt = " | Initializing..."
                        print(f"  Worker {wid+1}: Running {job_name} | Time {mins:02d}:{secs:02d}{ckpt_txt}")
                    else:
                        print(f"  Worker {wid+1}: Idle")
                print("------------------------------------------------------------\n")
                sys.stdout.flush()
                last_update_time = now

            i += 1
            _time.sleep(update_interval if github_mode else 0.2)

        # Final summary (only printed once)
        if github_mode:
            print("------------------------------------------------------------")
            print(
                f"✅ All jobs complete! Passed: {self.passed_jobs} ({(self.passed_jobs/total_jobs)*100:.1f}%) | "
                f"Failed: {self.failed_jobs} ({(self.failed_jobs/total_jobs)*100:.1f}%)"
            )
            print("------------------------------------------------------------")
            sys.stdout.flush()
        elif not self.silent_mode:
            sys.stdout.write(f"\033[{self.max_parallelism + 2}F")
            for wid in range(self.max_parallelism):
                sys.stdout.write(f"\033[KWorker {wid+1}: Idle\n")
            sys.stdout.write(
                f"\033[K✅  [████████████████████████████████████████] All jobs complete!  "
                f"Passed ✅: {self.passed_jobs} ({(self.passed_jobs/total_jobs)*100:.1f}%) | "
                f"Failed ❌: {self.failed_jobs} ({(self.failed_jobs/total_jobs)*100:.1f}%)\n\n"
            )
            sys.stdout.flush()


    def write_summary_results(self):
        """Write a summary results YAML file with just pass/fail counts."""
        with self.results_lock:
            failed_names = [name for name, r in self.results_data.items() if r.get("status") == "failed"]
            summary = {
                "total_jobs": len(self.results_data),
                "passed_jobs": sum(1 for r in self.results_data.values() if r.get("status") == "passed"),
                "failed_jobs": sum(1 for r in self.results_data.values() if r.get("status") == "failed"),
                "failed_list": failed_names,
            }
            os.makedirs(os.path.dirname(self.results_file_summary), exist_ok=True)
            tmp_path = self.results_file_summary + ".tmp"
            with open(tmp_path, "w") as f:
                yaml.dump(summary, f, default_flow_style=False)
            os.replace(tmp_path, self.results_file_summary)

        # Only print results summary in interactive mode (not GitHub)
        if not self.github_autotest_mode and not self.silent_mode:
            if failed_names:
                print("Failed jobs:")
                for n in failed_names:
                    print(f" - {n}")
            else:
                print("All jobs passed.")



    def record_result(self, job_name, success, elapsed_time):
        """Record job result and write to results file safely."""
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
        """Watches the log file for CHECKPOINT REACHED messages and updates job state."""
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
                        ts_str = m.group(3)
                        parsed_ts = None
                        if ts_str:
                            try:
                                from datetime import datetime
                                parsed_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                                import time as __t
                                parsed_ts = __t.mktime(parsed_dt.timetuple())
                            except Exception:
                                parsed_ts = None

                        now = _time.time()
                        with self.job_worker_lock:
                            initialized = self.job_initialized.get(job_name, False)

                            # Detect valid setup@0 and mark initialization
                            if (
                                step.lower() == "setup"
                                and it == 0
                                and parsed_ts is not None
                                and parsed_ts >= getattr(self, "regression_start_ts", 0)
                            ):
                                self.job_initialized[job_name] = True
                                initialized = True

                            # Ignore any checkpoint updates before initialization
                            if not initialized and not (
                                step.lower() == "setup" and it == 0
                            ):
                                continue

                            # Record checkpoint if initialized
                            data = self.job_checkpoints.setdefault(job_name, {"latest": None, "history": []})
                            data["latest"] = (step, it)
                            data["history"].append((step, it, parsed_ts if parsed_ts is not None else now))

                    last_size = f.tell()
                    last_mtime = mtime
            except Exception:
                pass

            _time.sleep(0.5)




if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Codesign Regression Run",
            description="Runs the codesign framework in regression mode. Return code will be 0 if all tests pass, 1 if any test fails.",
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

        parser.add_argument(
            "-q",
            "--quiet_mode",
            action="store_true",
            help="Run the regression in silent mode, suppressing output.",
        )

        parser.add_argument(
            "-g",
            "--github_autotest_mode",
            action="store_true",
            help="Displays the output in a format suitable for GitHub Actions annotations.",
        )

        parser.add_argument(
            "--preinstalled_openroad_path",
            type=str,
            help="Path to a pre-installed OpenROAD installation. This is primarily useful for CI testing where OpenROAD is pre-installed on the system.",
        )

        parser.add_argument(
            "--dsp_sweep",
            type=str,
            help="Sweep max_dsp parameter across a range. Format: 'start:end:step' (e.g., '10:2000:10'). Creates separate jobs for each DSP value.",
        )

        args = parser.parse_args()

        ## check if we are in the codesign directory
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(cwd, "src")) or not os.path.exists(os.path.join(cwd, "test")):
            print("Error: must be run from the codesign root directory")
            exit(1)

        ## make sure that if a file is specified in the -l command, it ends in .list.yaml
        if args.test_list is not None and not args.test_list.endswith(".list.yaml"):
            print("Error: Regression list file must end in .list.yaml")
            exit(1)

        reg_run = RegressionRun(cfg=None, codesign_root_dir=cwd, single_config_path=args.single_config, test_list_path=args.test_list, max_parallelism=args.max_parallelism, absolute_paths=args.absolute_paths, silent_mode=args.quiet_mode, github_autotest_mode=args.github_autotest_mode, preinstalled_openroad_path=args.preinstalled_openroad_path, dsp_sweep=args.dsp_sweep)

        regression_instance = reg_run

        exit_code = reg_run.run_regression()

        exit(exit_code)
