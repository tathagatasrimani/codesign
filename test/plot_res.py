import argparse
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import csv

def extract_execution_time_and_edp(log_path):
    """Extract execution_time and EDP from run_codesign.log"""
    exec_time = None
    edp = None
    
    try:
        with open(log_path, "r") as f:
            content = f.read()
            
            # Look for "Execution time: <number>"
            match = re.search(r"Execution time:\s*([0-9]*\.?[0-9]+)", content)
            if match:
                exec_time = float(match.group(1))
            
            # Look for "edp: <number>" in the log
            match = re.search(r"edp:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", content)
            if match:
                edp = float(match.group(1))
    except FileNotFoundError:
        pass
    
    return exec_time, edp

def parse_dsp_from_dirname(dirname):
    """Extract DSP value from directory name like 'benchmark_3mm_test_auto_no_wires_dsp100'"""
    match = re.search(r"_dsp(\d+)$", dirname)
    if match:
        return int(match.group(1))
    return None

def extract_actual_dsp_from_csv(entry_path, benchmark_name, max_dsp):
    """
    Extract actual DSP used from kernel_3mm_space.csv.
    Finds the max DSP value that is <= max_dsp.
    """
    # Extract the kernel name (e.g., "kernel_3mm" from "benchmark_3mm_test_auto_no_wires")
    # Pattern: benchmark_<kernel_name>_test_auto_no_wires
    match = re.search(r"benchmark_(.+?)_test", benchmark_name)
    if match:
        kernel_name = match.group(1)
    else:
        # Fallback: just remove "benchmark_" prefix
        kernel_name = benchmark_name.replace('benchmark_', '')
    
    # Construct path to CSV file
    csv_path = os.path.join(
        entry_path, 
        "tmp", 
        f"tmp_kernel_{kernel_name}_edp_0",
        "benchmark_setup",
        "function_hier_output",
        f"kernel_{kernel_name}_space.csv"
    )
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}")
        return None
    
    try:
        dsp_values = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # Find the 'dsp' column (second to last)
            dsp_idx = len(header) - 2
            
            for row in reader:
                if len(row) > dsp_idx:
                    dsp_val = int(row[dsp_idx])
                    dsp_values.append(dsp_val)
        
        # Filter DSP values <= max_dsp and return the maximum
        valid_dsps = [d for d in dsp_values if d <= max_dsp]
        if valid_dsps:
            return max(valid_dsps)
        
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
    
    return None

def scan_directory(root_dir):
    """
    Scan root_dir for subdirectories, extract DSP values, execution times, and EDP.
    Returns: list of (actual_dsp_used, execution_time, edp, benchmark_name) tuples
    """
    results = []
    
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory")
        return results
    
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        
        if not os.path.isdir(entry_path):
            continue
        
        # Extract max_dsp value from directory name
        max_dsp = parse_dsp_from_dirname(entry)
        if max_dsp is None:
            continue
        
        # Extract benchmark name (everything before _dsp)
        benchmark = entry.rsplit("_dsp", 1)[0]
        
        # Look for run_codesign.log
        log_path = os.path.join(entry_path, "run_codesign.log")
        exec_time, edp = extract_execution_time_and_edp(log_path)
        
        if exec_time is None or edp is None:
            print(f"Warning: Could not extract execution_time or EDP from {entry}")
            continue
        
        # Extract actual DSP used from CSV
        actual_dsp_used = extract_actual_dsp_from_csv(entry_path, benchmark, max_dsp)
        
        if actual_dsp_used is None:
            print(f"Warning: Could not extract actual DSP from CSV for {entry}")
            continue
        
        results.append((actual_dsp_used, exec_time, edp, benchmark))
        print(f"Found: {entry} -> Max DSP: {max_dsp}, Actual DSP Used: {actual_dsp_used}, Exec Time: {exec_time:.2f}, EDP: {edp:.2e}")
    
    return results

def group_by_benchmark(results):
    """Group results by benchmark name"""
    grouped = defaultdict(list)
    for dsp, exec_time, edp, benchmark in results:
        grouped[benchmark].append((dsp, exec_time, edp))
    
    # Sort by DSP value
    for b in grouped:
        grouped[b].sort(key=lambda x: x[0])
    
    return grouped

def plot_results(grouped_data, output_dir):
    """Plot execution time vs actual DSP used for each benchmark"""
    for benchmark, data in grouped_data.items():
        dsps = [d[0] for d in data]
        times = [d[1] for d in data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(dsps, times, marker="o", linewidth=2, markersize=6)
        plt.title(f"{benchmark} — Actual DSP Used vs Execution Time")
        plt.xlabel("Actual DSP Used (from CSV)")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join(output_dir, f"{benchmark}_actual_dsp_vs_time.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

def plot_edp_results(grouped_data, output_dir):
    """Plot EDP vs actual DSP used for each benchmark"""
    for benchmark, data in grouped_data.items():
        dsps = [d[0] for d in data]
        edps = [d[2] for d in data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(dsps, edps, marker="s", linewidth=2, markersize=6, color='red')
        plt.title(f"{benchmark} — Actual DSP Used vs EDP")
        plt.xlabel("Actual DSP Used (from CSV)")
        plt.ylabel("EDP (Energy-Delay Product)")
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        out_path = os.path.join(output_dir, f"{benchmark}_EDP_plot.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved EDP plot: {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Parse DSP sweep results and extract execution times and EDP"
    )
    ap.add_argument(
        "--path",
        required=True,
        help="Path to directory containing sweep results (e.g., test/regression_results/yarch_sweep_1.list/yarch_sweep)"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for each benchmark"
    )
    args = ap.parse_args()
    
    print(f"Scanning directory: {args.path}\n")
    results = scan_directory(args.path)
    
    if not results:
        print("No results found!")
        return
    
    grouped = group_by_benchmark(results)
    
    print("\n" + "="*80)
    print("SUMMARY: DSP → (Execution Time, EDP) Tuples")
    print("="*80)
    
    for benchmark, data in grouped.items():
        print(f"\n{benchmark}:")
        for dsp_val, exec_time, edp in data:
            print(f"  DSP {dsp_val:5d}: Exec Time = {exec_time:15.2f}, EDP = {edp:.4e}")
    
    if args.plot:
        output_dir = os.path.dirname(args.path)
        print(f"\nGenerating plots in {output_dir}...")
        plot_results(grouped, output_dir)
        plot_edp_results(grouped, output_dir)

if __name__ == "__main__":
    main()