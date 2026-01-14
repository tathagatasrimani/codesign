#!/usr/bin/env python3

import os
import sys
import yaml
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

class ScaleHLSSweepAutomation:
    def __init__(self, config_file: str = None, base_dir: str = None):
        # If no base_dir provided, use the directory where this script is located
        if base_dir is None:
            base_dir = str(Path(__file__).parent)

        self.base_dir = Path(base_dir)
        self.regression_dir = self.base_dir / "regressions/auto_tests/yarch_exps"
        
        # If no config_file provided, use default location
        if config_file is None:
            config_file = str(self.regression_dir / "configs/kernel_sweeps.yaml")
        
        self.config_file = config_file
        self.results = []
        
        print(f"Base directory set to: {self.base_dir}")
        print(f"Config file: {self.config_file}\n")
        
        self.load_config()

    def load_config(self):
        """Load kernel configuration from YAML"""
        if not os.path.exists(self.config_file):
            print(f"ERROR: Config file not found: {self.config_file}")
            sys.exit(1)
            
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.kernels = self.config.get('kernels', [])

    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70 + "\n")

    def print_status(self, kernel_name: str, step: str, status: str):
        """Print status with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_code = "✓" if status == "DONE" else "►" if status == "RUNNING" else "✗"
        print(f"[{timestamp}] {color_code} {kernel_name:20} | {step:25} | {status}")

    def create_experiment_folder(self, kernel_name: str) -> Path:
        """Create experiment folder if it doesn't exist"""
        exp_folder = self.regression_dir / f"{kernel_name}_exp"
        exp_folder.mkdir(parents=True, exist_ok=True)
        self.print_status(kernel_name, "Create folder", "DONE")
        return exp_folder

    def create_save_config(self, kernel_name: str, exp_folder: Path, config_params: Dict):
        """Create yarch_{kernel_name}_save.yaml"""
        save_config = {
            f"benchmark_{kernel_name}_save": {
                "base_cfg": "vitis_test",
                "args": {
                    "benchmark": kernel_name,
                    "obj": config_params.get("obj", "delay"),
                    "workload_size": config_params.get("workload_size", 32),
                    "checkpoint_save_dir": f"tmp_{kernel_name}",
                    "stop_at_checkpoint": "setup",
                    "max_iter_num_scalehls": config_params.get("max_iter_num_scalehls", 10)
                }
            }
        }

        save_file = exp_folder / f"yarch_{kernel_name}_save.yaml"
        with open(save_file, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)
        self.print_status(kernel_name, "Create save config", "DONE")
        return save_file

    def create_sweep_list(self, kernel_name: str, exp_folder: Path):
        """Create yarch_sweep_{kernel_name}_exp.list.yaml"""
        sweep_list = [f"yarch_{kernel_name}_save.yaml"]
        
        sweep_file = exp_folder / f"yarch_sweep_{kernel_name}_exp.list.yaml"
        with open(sweep_file, 'w') as f:
            yaml.dump(sweep_list, f, default_flow_style=False)
        self.print_status(kernel_name, "Create sweep list", "DONE")
        return sweep_file

    def create_load_configs(self, kernel_name: str, exp_folder: Path, config_params: Dict):
        """Create load configuration files"""
        
        # Base load config template
        base_load_args = {
            "benchmark": kernel_name,
            "checkpoint_load_dir": f"tmp_{kernel_name}",
            "checkpoint_start_step": "setup",
            "stop_at_checkpoint": "pd",
            "max_dsp": config_params.get("max_dsp", 100),
            "max_iter_num_scalehls": config_params.get("max_iter_num_scalehls", 10)
        }

        # No wires config
        no_wires_config = {
            f"benchmark_{kernel_name}_test_auto_no_wires": {
                "base_cfg": "vitis_test",
                "args": {
                    **base_load_args,
                    "zero_wirelength_costs": True,
                    "ignore_wirelength_costs": True
                }
            }
        }

        # With wires config
        with_wires_config = {
            f"benchmark_{kernel_name}_test_auto_with_wires": {
                "base_cfg": "vitis_test",
                "args": base_load_args.copy()
            }
        }

        # Constant wires config
        constant_wires_config = {
            f"benchmark_{kernel_name}_test_auto_constant_wires": {
                "base_cfg": "vitis_test",
                "args": {
                    **base_load_args,
                    "constant_wire_length_cost": config_params.get("constant_wire_length_cost", 0.04)
                }
            }
        }

        # Write all configs
        configs = {
            "no_wires": ("yarch_load_sweep_no_wires.yaml", no_wires_config),
            "with_wires": ("yarch_sweep_load_with_wires.yaml", with_wires_config),
            "constant_wires": ("yarch_load_sweep_constant_wires.yaml", constant_wires_config)
        }

        files = []
        for config_type, (filename, config) in configs.items():
            filepath = exp_folder / filename
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            files.append(filename)

        self.print_status(kernel_name, "Create load configs", "DONE")
        return files

    def create_load_list(self, kernel_name: str, exp_folder: Path, exp_num: int):
        """Create yarch_sweep_exp_{i}_load_list.yaml"""
        load_list = [
            "yarch_load_sweep_no_wires.yaml",
            "yarch_sweep_load_with_wires.yaml",
            "yarch_load_sweep_constant_wires.yaml"
        ]

        list_file = exp_folder / f"yarch_sweep_exp{exp_num}_load_list.yaml"
        with open(list_file, 'w') as f:
            yaml.dump(load_list, f, default_flow_style=False)
        self.print_status(kernel_name, "Create load list", "DONE")
        return list_file

    def process_kernel(self, kernel_name: str, kernel_config: Dict, exp_num: int):
        """Process a single kernel through the entire pipeline"""
        self.print_header(f"Processing Kernel: {kernel_name}")

        # Create experiment folder
        exp_folder = self.create_experiment_folder(kernel_name)

        # Create save configuration
        self.create_save_config(kernel_name, exp_folder, kernel_config)

        # Create sweep list
        self.create_sweep_list(kernel_name, exp_folder)

        # Create load configurations
        self.create_load_configs(kernel_name, exp_folder, kernel_config)

        # Create load list
        self.create_load_list(kernel_name, exp_folder, exp_num)

        self.results.append({"kernel": kernel_name, "status": "SUCCESS"})
        self.print_status(kernel_name, "All files created", "DONE")

    def print_summary(self):
        """Print execution summary"""
        self.print_header("Execution Summary")
        
        total = len(self.results)
        
        print("Files and directories created for the following kernels:\n")
        for result in self.results:
            status_symbol = "✓" if result["status"] == "SUCCESS" else "✗"
            print(f"{status_symbol} {result['kernel']:20} | {result['status']}")
        
        print(f"\nTotal: {total} | Passed: {sum(1 for r in self.results if r['status'] == 'SUCCESS')} | Failed: {sum(1 for r in self.results if r['status'] == 'FAILED')}")
        print("="*70 + "\n")

    def run_all(self, dsp_sweep: str = "128:1024:8"):
        """Run the entire automation pipeline"""
        self.print_header("ScaleHLS Sweep Automation Started")
        
        for idx, kernel in enumerate(self.kernels, 1):
            kernel_name = kernel.get("name")
            kernel_config = {k: v for k, v in kernel.items() if k != "name"}
            
            self.process_kernel(kernel_name, kernel_config, idx)

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="ScaleHLS Sweep Automation Script")
    parser.add_argument("-c", "--config", default=None, help="YAML config file (default: regressions/auto_tests/yarch_exps/configs/kernel_sweeps.yaml)")
    parser.add_argument("-d", "--dsp_sweep", default="128:1024:8", help="DSP sweep parameters (default: 128:1024:8)")
    
    args = parser.parse_args()

    automation = ScaleHLSSweepAutomation(args.config)
    automation.run_all(args.dsp_sweep)


if __name__ == "__main__":
    main()