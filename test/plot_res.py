import argparse
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import yaml
import math

def extract_execution_time_and_edp(log_path, debug=False):
    """Extract execution_time and EDP from run_codesign.log"""

    exec_time = None
    edp = None

    if debug:
        print(f"Extracting execution_time and EDP from {log_path}")
    
    try:
        with open(log_path, "r") as f:
            content = f.read()
            
            # Look for "edp: <number>, sub expressions: {...}" pattern
            match = re.search(r"edp:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?),\s*sub expressions:\s*(\{.+?\})", content, re.DOTALL)
            if match:
                edp = float(match.group(1))
                sub_expr_str = match.group(2)
                
                # Parse the dictionary string to extract execution_time
                exec_time_match = re.search(r"'execution_time':\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", sub_expr_str)
                if exec_time_match:
                    exec_time = float(exec_time_match.group(1))
    except FileNotFoundError:
        pass
    
    return exec_time, edp

def parse_dsp_from_dirname(dirname, debug=False):
    """Extract DSP value from directory name like 'benchmark_3mm_test_auto_no_wires_dsp100'"""
    if debug:
        print(f"Parsing DSP from directory name: {dirname}")
    match = re.search(r"_dsp(\d+)$", dirname)
    if match:
        if debug:
            print(f"Extracted DSP value: {match.group(1)}")
        return int(match.group(1))
    
    if debug:
        print(f"Could not extract DSP value from directory name: {dirname}")
    return None

def extract_actual_dsp_from_csv(entry_path, benchmark_name, max_dsp, kernel=None):
    """
    Extract actual DSP used from kernel_3mm_space.csv.
    Finds the max DSP value that is <= max_dsp.
    """
    # print(f"Extracting actual DSP from CSV for benchmark: {benchmark_name} with max_dsp: {max_dsp}")
    # Extract the kernel name from pattern: benchmark_{kernel_name}_test_something
    match = re.search(r"benchmark_(.+?)_test_*", benchmark_name)
    if match:
        kernel_name = match.group(1)
    else:
        # print(f"Warning: Could not parse kernel name from {benchmark_name}")
        return None

    # print(f"Derived kernel name: {kernel_name}")
    
    # Construct path to CSV file
    append_kernel = f"kernel_{kernel_name}" if kernel else f"{kernel_name}"
    # print(f"Using append_kernel: {append_kernel}")
    csv_path = os.path.join(
        entry_path, 
        "tmp", 
        f"tmp_{append_kernel}_edp_0",
        "benchmark_setup",
        "function_hier_output",
        f"{append_kernel}_space.csv"
    )

    # print(f"Looking for DSP CSV at: {csv_path}")
    
    if not os.path.exists(csv_path):
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
        pass
    
    return None

def get_sweep_label(directory_name, debug=False):
    """Convert directory name to sweep label."""
    lower = directory_name.lower()
    if debug:
        print(f"Generating sweep label for directory: {directory_name}")
    # Match no-wires first to avoid the "with_wires" substring trap (e.g., "without_wires")
    if "no_wires" in lower or "without_wires" in lower:
        if debug:
            print("Detected no wires in directory name.")
        return "No wires"
    elif "constant_wires" in lower or "fixed_wires" in lower:
        if debug:
            print("Detected fixed wire cost in directory name.")
        return "Fixed wire cost"
    elif "with_wires" in lower:
        if debug:
            print("Detected wires cost estimated in directory name.")
        return "Wires cost estimated"
    else:
        return "No wires"

def normalize_benchmark_name(benchmark):
    """Extract kernel_name from benchmark_{kernel_name}_test_something."""
    m = re.search(r'^benchmark_(.+?)_test', benchmark)
    if m:
        return m.group(1)
    return benchmark

def scan_directory(root_dir, sweep_label, kernel=None, debug=False):
    """
    Scan root_dir for subdirectories, extract DSP values, execution times, and EDP.
    Returns: list of (actual_dsp_used, execution_time, edp, benchmark_name, sweep_label, norm_benchmark)
    """
    results = []
    
    if not os.path.isdir(root_dir):
        if debug:
            print(f"Error: {root_dir} is not a valid directory")
        return results
    
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        
        if not os.path.isdir(entry_path):
            continue
        
        # Extract max_dsp value from directory name
        max_dsp = parse_dsp_from_dirname(entry, debug=debug)
        if max_dsp is None:
            if debug:
                print(f"[{sweep_label}] Skipping {entry}: could not parse max_dsp from directory name")
            continue
        
        # Extract benchmark name (everything before _dsp)
        benchmark = entry.rsplit("_dsp", 1)[0]

        norm_benchmark = normalize_benchmark_name(benchmark)
        
        if debug:
            print(f"[{sweep_label}] Scanning {entry} (max_dsp={max_dsp})...")
        
        # Look for run_codesign.log
        log_path = os.path.join(entry_path, "run_codesign.log")
        exec_time, edp = extract_execution_time_and_edp(log_path, debug=debug)
        
        if exec_time is None or edp is None:
            if debug:
                print(f"[{sweep_label}] Skipping {entry}: missing exec_time or edp")
            continue
        
        # Extract actual DSP used from CSV using the benchmark name
        actual_dsp_used = extract_actual_dsp_from_csv(entry_path, benchmark, max_dsp, kernel=kernel)
        
        if actual_dsp_used is None:
            if debug:
                print(f"[{sweep_label}] Skipping {entry}: could not extract actual DSP from CSV")
            continue
        
        results.append((actual_dsp_used, exec_time, edp, benchmark, sweep_label, norm_benchmark))
        print(f"[{sweep_label}] Found: {entry} -> DSP: {actual_dsp_used}, Exec Time: {exec_time:.2f}, EDP: {edp:.2e}")
    
    return results

def group_by_benchmark_and_sweep(results):
    """Group results by normalized benchmark and sweep label, keeping the lowest exec time per DSP."""
    dedup = defaultdict(lambda: defaultdict(dict))
    for dsp, exec_time, edp, benchmark, sweep_label, norm_bench in results:
        current = dedup[norm_bench][sweep_label].get(dsp)
        if current is None or exec_time < current[0]:
            dedup[norm_bench][sweep_label][dsp] = (exec_time, edp)

    grouped = defaultdict(lambda: defaultdict(list))
    for norm_bench, sweeps in dedup.items():
        for sweep_label, dsp_map in sweeps.items():
            for dsp, (exec_time, edp) in sorted(dsp_map.items()):
                grouped[norm_bench][sweep_label].append((dsp, exec_time, edp))

    return grouped

def plot_execution_time_results(grouped_data, output_dir, debug=False):
    """Single plot per normalized benchmark: Execution time vs DSP with all sweep curves"""
    for norm_bench, sweep_data in grouped_data.items():
        plt.figure(figsize=(10, 6))

        colors = {
            'No wires': '#1f77b4',
            'Wires cost estimated': '#ff7f0e',
            'Fixed wire cost': '#2ca02c'
        }
        markers = {
            'No wires': 'o',
            'Wires cost estimated': 's',
            'Fixed wire cost': '^'
        }

        # Find the minimum execution time across all sweeps
        all_times = []
        for data in sweep_data.values():
            for d in data:
                all_times.append(d[1])
        
        min_time = min(all_times) if all_times else 0
        
        if debug:
            print(f"\n[PLOT] Benchmark: {norm_bench}")
            print(f"[PLOT] Minimum execution time: {min_time}")
        
        # Plot with normalized times (difference from minimum)
        for sweep_label, data in sorted(sweep_data.items()):
            if len(data) == 0:
                continue
                
            dsps = [d[0] for d in data]
            times = [d[1] - min_time for d in data]  # Subtract minimum from all times

            if debug:
                print(f"[PLOT] {sweep_label}:")
                for dsp, time_diff, orig_time in zip(dsps, times, [d[1] for d in data]):
                    print(f"[PLOT]   DSP={dsp}, Original Time={orig_time}, Normalized Time={time_diff}")

            plt.plot(
                dsps,
                times,
                marker=markers.get(sweep_label, 'o'),
                linewidth=2.5,
                markersize=8,
                label=sweep_label,
                color=colors.get(sweep_label)
            )

        plt.title(f"{norm_bench} — DSP vs Execution Time (normalized)", fontsize=14, fontweight='bold')
        plt.xlabel("Actual DSP Used", fontsize=12)
        plt.ylabel("Execution Time Difference (seconds)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')

        out_path = os.path.join(output_dir, f"{norm_bench}_dsp_vs_time.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        if debug:
            print(f"[PLOT] Saved plot: {out_path}\n")

def plot_execution_time_results_log_scale(grouped_data, output_dir, debug=False):
    """Single plot per normalized benchmark: Execution time vs DSP with all sweep curves (log scale)"""
    for norm_bench, sweep_data in grouped_data.items():
        plt.figure(figsize=(10, 6))

        colors = {
            'No wires': '#1f77b4',
            'Wires cost estimated': '#ff7f0e',
            'Fixed wire cost': '#2ca02c'
        }
        markers = {
            'No wires': 'o',
            'Wires cost estimated': 's',
            'Fixed wire cost': '^'
        }

        if debug:
            print(f"\n[PLOT LOG] Benchmark: {norm_bench}")
        
        # Plot with log scale on y-axis
        for sweep_label, data in sorted(sweep_data.items()):
            if len(data) == 0:
                continue
                
            dsps = [d[0] for d in data]
            times = [math.log(d[1]) for d in data]

            times_min = min(times)
            times_max = max(times)
            if debug:
                print(f"[PLOT LOG] {sweep_label} - Min log time: {times_min}, Max log time: {times_max}")

            times = [t - times_min for t in times]  # Normalize by subtracting min log time

            times = [t/times_max for t in times]  # Scale to [0, 1]

            if debug:
                print(f"[PLOT LOG] {sweep_label}:")
                for dsp, orig_time in zip(dsps, times):
                    print(f"[PLOT LOG]   DSP={dsp}, Execution Time={orig_time}")

            plt.plot(
                dsps,
                times,
                marker=markers.get(sweep_label, 'o'),
                linewidth=2.5,
                markersize=8,
                label=sweep_label,
                color=colors.get(sweep_label)
            )

        plt.title(f"{norm_bench} — DSP vs Execution Time (log scale)", fontsize=14, fontweight='bold')
        plt.xlabel("Actual DSP Used", fontsize=12)
        plt.ylabel("Execution Time (seconds, log scale)", fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which='both')
        plt.legend(fontsize=11, loc='best')

        out_path = os.path.join(output_dir, f"{norm_bench}_dsp_vs_delay_log.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        if debug:
            print(f"[PLOT LOG] Saved plot: {out_path}\n")

def plot_edp_results(grouped_data, output_dir):
    """Plot EDP vs actual DSP used for each benchmark with multiple sweeps"""
    for benchmark, sweep_data in grouped_data.items():
        plt.figure(figsize=(12, 7))
        
        colors = {'No wires': '#1f77b4', 'Wires cost estimated': '#ff7f0e', 'Fixed wire cost': '#2ca02c'}
        markers = {'No wires': 'o', 'Wires cost estimated': 's', 'Fixed wire cost': '^'}
        
        for sweep_label, data in sorted(sweep_data.items()):
            dsps = [d[0] for d in data]
            edps = [d[2] for d in data]
            
            color = colors.get(sweep_label, None)
            marker = markers.get(sweep_label, 'o')
            
            plt.plot(dsps, edps, marker=marker, linewidth=2, markersize=7, 
                    label=sweep_label, color=color)
        
        plt.title(f"{benchmark} — DSP vs EDP", fontsize=14, fontweight='bold')
        plt.xlabel("Actual DSP Used", fontsize=12)
        plt.ylabel("EDP (Energy-Delay Product)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        out_path = os.path.join(output_dir, f"{benchmark}_EDP_plot.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved EDP plot: {out_path}")

def plot_edp_deviation_results(grouped_data, output_dir):
    """
    For each benchmark, generate deviation plots using each sweep as baseline.
    Plots % EDP deviation vs actual DSP used. The baseline sweep is shown as a
    zero line; other sweeps are plotted when DSP points match the baseline.
    """
    colors = {'No wires': '#1f77b4', 'Wires cost estimated': '#ff7f0e', 'Fixed wire cost': '#2ca02c'}
    markers = {'No wires': 'o', 'Wires cost estimated': 's', 'Fixed wire cost': '^'}
    baselines = ["Wires cost estimated", "Fixed wire cost", "No wires"]

    for benchmark, sweep_data in grouped_data.items():
        for baseline_label in baselines:
            baseline = sweep_data.get(baseline_label)
            if not baseline:
                print(f"Skipping deviation plot for {benchmark}: no {baseline_label} baseline.")
                continue

            baseline_map = {dsp: edp for dsp, _, edp in baseline}
            baseline_dsps = sorted(baseline_map.keys())
            if not baseline_dsps:
                print(f"Skipping deviation plot for {benchmark}: empty {baseline_label} baseline.")
                continue

            plt.figure(figsize=(12, 7))

            # Zero line for baseline
            plt.plot(
                baseline_dsps,
                [0.0] * len(baseline_dsps),
                linestyle='--',
                color=colors.get(baseline_label, '#888888'),
                label=f"{baseline_label} (baseline)",
                linewidth=1.8,
            )

            for sweep_label, data in sweep_data.items():
                if sweep_label == baseline_label:
                    continue
                if not data:
                    continue
                sweep_map = {dsp: edp for dsp, _, edp in data}

                dsps = []
                deviations = []
                for dsp, edp_base in baseline_map.items():
                    if dsp not in sweep_map:
                        continue
                    edp_other = sweep_map[dsp]
                    dev_pct = (edp_other - edp_base) / edp_base * 100.0
                    dsps.append(dsp)
                    deviations.append(dev_pct)

                if not dsps:
                    continue

                plt.plot(
                    dsps,
                    deviations,
                    marker=markers.get(sweep_label, 'o'),
                    linewidth=2,
                    markersize=7,
                    label=sweep_label,
                    color=colors.get(sweep_label),
                )

            plt.title(f"{benchmark} — % EDP deviation vs {baseline_label}", fontsize=14, fontweight='bold')
            plt.xlabel("Actual DSP Used", fontsize=12)
            plt.ylabel(f"% EDP Deviation (relative to {baseline_label})", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11, loc='best')

            # Save with baseline in filename
            slug = baseline_label.lower().replace(" ", "_")
            out_path = os.path.join(output_dir, f"{benchmark}_EDP_deviation_vs_{slug}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved EDP deviation plot: {out_path}")

def plot_relative_time_delay_results(grouped_data, output_dir, debug=False):
    """
    For each benchmark, generate relative time delay plots using "Wires cost estimated" as baseline.
    Plots % execution time deviation vs actual DSP used. The baseline sweep is shown as a
    zero line; other sweeps are plotted when DSP points match the baseline.
    """
    colors = {'No wires': '#1f77b4', 'Wires cost estimated': '#ff7f0e', 'Fixed wire cost': '#2ca02c'}
    markers = {'No wires': 'o', 'Wires cost estimated': 's', 'Fixed wire cost': '^'}
    baseline_label = "Wires cost estimated"

    for benchmark, sweep_data in grouped_data.items():
        baseline = sweep_data.get(baseline_label)
        if not baseline:
            print(f"Skipping relative time delay plot for {benchmark}: no {baseline_label} baseline.")
            continue

        baseline_map = {dsp: exec_time for dsp, exec_time, _ in baseline}
        baseline_dsps = sorted(baseline_map.keys())
        if not baseline_dsps:
            print(f"Skipping relative time delay plot for {benchmark}: empty {baseline_label} baseline.")
            continue

        plt.figure(figsize=(12, 7))

        # Zero line for baseline
        plt.plot(
            baseline_dsps,
            [0.0] * len(baseline_dsps),
            linestyle='--',
            color=colors.get(baseline_label, '#888888'),
            label=f"{baseline_label} (baseline)",
            linewidth=1.8,
        )

        for sweep_label, data in sweep_data.items():
            if sweep_label == baseline_label:
                continue
            if not data:
                continue
            
            sweep_map = {dsp: exec_time for dsp, exec_time, _ in data}

            dsps = []
            deviations = []
            for dsp, exec_time_base in baseline_map.items():
                if dsp not in sweep_map:
                    continue
                exec_time_other = sweep_map[dsp]
                dev_pct = (exec_time_other - exec_time_base) / exec_time_base * 100.0
                dsps.append(dsp)
                deviations.append(dev_pct)

            if not dsps:
                continue

            plt.plot(
                dsps,
                deviations,
                marker=markers.get(sweep_label, 'o'),
                linewidth=2.5,
                markersize=8,
                label=sweep_label,
                color=colors.get(sweep_label),
            )

        plt.title(f"{benchmark} — Relative Time Delay vs {baseline_label}", fontsize=14, fontweight='bold')
        plt.xlabel("Actual DSP Used", fontsize=12)
        plt.ylabel(f"% Execution Time Deviation (relative to {baseline_label})", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')

        out_path = os.path.join(output_dir, f"{benchmark}_relative_time_delay_graph.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[RELATIVE TIME DELAY] Saved plot: {out_path}\n")

def plot_final_deviation_plot(grouped_data, output_dir, debug=False):
    """
    For each benchmark:
    - "Wires cost estimated" at DSP n uses baseline from DSP (n-1) within same sweep
    - Other sweeps ("No wires", "Fixed wire cost") at DSP n use baseline from "Wires cost estimated" at DSP n
    """
    colors = {'No wires': '#1f77b4', 'Wires cost estimated': '#ff7f0e', 'Fixed wire cost': '#2ca02c'}
    markers = {'No wires': 'o', 'Wires cost estimated': 's', 'Fixed wire cost': '^'}
    baseline_label = "Wires cost estimated"

    for benchmark, sweep_data in grouped_data.items():
        baseline = sweep_data.get(baseline_label)
        if not baseline:
            if debug:
                print(f"Skipping final_deviation_plot for {benchmark}: no {baseline_label} baseline.")
            continue

        # Create a map: DSP -> baseline execution time
        baseline_data = sorted(baseline, key=lambda x: x[0])  # Sort by DSP
        baseline_map = {dsp: exec_time for dsp, exec_time, _ in baseline_data}
        baseline_dsps = sorted(baseline_map.keys())
        if not baseline_dsps:
            if debug:
                print(f"Skipping final_deviation_plot for {benchmark}: empty {baseline_label} baseline.")
            continue

        plt.figure(figsize=(12, 7))

        if debug:
            print(f"\n[FINAL DEVIATION] Benchmark: {benchmark}")

        for sweep_label, data in sorted(sweep_data.items()):
            if not data:
                continue

            sweep_data_sorted = sorted(data, key=lambda x: x[0])  # Sort by DSP
            sweep_map = {dsp: exec_time for dsp, exec_time, _ in sweep_data_sorted}

            dsps = []
            deviations = []

            if sweep_label == baseline_label:
                # For "Wires cost estimated": baseline is the previous DSP point in same sweep
                for i, (dsp, exec_time_current, _) in enumerate(sweep_data_sorted):
                    if i == 0:
                        # First point has no previous baseline, use 0% deviation
                        dev_pct = 0.0
                    else:
                        # Use previous DSP point in same sweep as baseline
                        prev_dsp, exec_time_prev, _ = sweep_data_sorted[i - 1]
                        dev_pct = (exec_time_current - exec_time_prev) / exec_time_prev * 100.0
                    
                    dsps.append(dsp)
                    deviations.append(dev_pct)
                    if debug:
                        print(f"[FINAL DEVIATION] {sweep_label} DSP={dsp}: Deviation={dev_pct:.2f}% "
                              f"(from prev point)")
            else:
                # For other sweeps: baseline is "Wires cost estimated" at same DSP
                for dsp, exec_time_sweep, _ in sweep_data_sorted:
                    if dsp in baseline_map:
                        exec_time_baseline = baseline_map[dsp]
                        dev_pct = (exec_time_sweep - exec_time_baseline) / exec_time_baseline * 100.0
                        dsps.append(dsp)
                        deviations.append(dev_pct)
                        if debug:
                            print(f"[FINAL DEVIATION] {sweep_label} DSP={dsp}: Deviation={dev_pct:.2f}% "
                                  f"(from {baseline_label} at same DSP)")

            if not dsps:
                continue

            # Sort by DSP for plotting
            sorted_pairs = sorted(zip(dsps, deviations), key=lambda x: x[0])
            dsps_sorted = [p[0] for p in sorted_pairs]
            deviations_sorted = [p[1] for p in sorted_pairs]

            plt.plot(
                dsps_sorted,
                deviations_sorted,
                marker=markers.get(sweep_label, 'o'),
                linewidth=2.5,
                markersize=8,
                label=sweep_label,
                color=colors.get(sweep_label),
            )

        # Zero line
        plt.axhline(0.0, linestyle='--', color='#888888', linewidth=1.5, label="0% baseline")

        plt.title(f"{benchmark} — Execution Time Deviation", fontsize=14, fontweight='bold')
        plt.xlabel("Actual DSP Used", fontsize=12)
        plt.ylabel("% Execution Time Deviation", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')

        out_path = os.path.join(output_dir, f"{benchmark}_final_deviation_plot.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        if debug:
            print(f"[FINAL DEVIATION] Saved plot: {out_path}\n")

def main():
    ap = argparse.ArgumentParser(
        description="Parse multiple DSP sweep results and generate comparison plots"
    )
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Path to regression_results directory containing sweep subdirs (e.g., test/regression_results/yarch_sweep_exp1.list)"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for each benchmark"
    )
    ap.add_argument(
        "--kernel",
        type=str,
        help="Tells if the the benchmark is kernel or not."
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    args = ap.parse_args()
    
    results_dir = args.results_dir
    
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a valid directory")
        return
    
    # Scan all subdirectories in results_dir
    all_results = []
    
    for entry in sorted(os.listdir(results_dir)):
        sweep_path = os.path.join(results_dir, entry)
        if not os.path.isdir(sweep_path):
            continue
        
        sweep_label = get_sweep_label(entry, debug=args.debug)
        
        if args.debug:
            print(f"\nScanning {sweep_label} ({entry})...")
        results = scan_directory(sweep_path, sweep_label, kernel=args.kernel, debug=args.debug)
        if args.debug:
            print(f"Found {len(results)} valid results in {entry}.")
        all_results.extend(results)
    
    if not all_results:
        print("No results found!")
        return
    
    grouped = group_by_benchmark_and_sweep(all_results)
    
    print("\n" + "="*80)
    print("SUMMARY: Results grouped by Benchmark and Sweep Type")
    print("="*80)
    
    for benchmark, sweep_data in grouped.items():
        print(f"\n{benchmark}:")
        for sweep_label, data in sorted(sweep_data.items()):
            print(f"  {sweep_label}:")
            for dsp_val, exec_time, edp in data:
                print(f"    DSP {dsp_val:5d}: Exec Time = {exec_time:15.2f}, EDP = {edp:.4e}")
    
    if args.plot:
        output_dir = results_dir
        print(f"\nGenerating comparison plots in {output_dir}...")
        plot_execution_time_results(grouped, output_dir, debug=args.debug)
        plot_execution_time_results_log_scale(grouped, output_dir, debug=args.debug)
        plot_relative_time_delay_results(grouped, output_dir, debug=args.debug)
        plot_final_deviation_plot(grouped, output_dir, debug=args.debug)
        plot_edp_results(grouped, output_dir)
        plot_edp_deviation_results(grouped, output_dir)

if __name__ == "__main__":
    main()