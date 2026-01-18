import os
import re
import argparse
import json
import csv

def count_ops(mlir_text: str) -> int:
    """
    Count MLIR operations by parsing the text.
    Counts lines that contain operation patterns (lines with '=', function calls, or specific MLIR ops)
    """
    count = 0
    lines = mlir_text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, comments, and pure structural elements
        if not line or line.startswith('//') or line.startswith('#'):
            continue
        
        # Count lines that look like operations:
        # - Lines with '=' (assignment operations)
        # - Lines ending with specific MLIR constructs
        # - Function definitions, blocks, etc.
        if ('=' in line and not line.startswith('module') and not line.startswith('func')) or \
           any(op in line for op in ['arith.', 'affine.', 'memref.', 'scf.', 'linalg.', 'func.call', 'func.return']):
            count += 1
    
    return count


def extract_delay_from_log(log_file_path: str, debug: bool = False):
    """
    Extract delay, clk_period, and energy from run_codesign.log file.
    Returns a tuple (execution_time, clk_period, delay_cycles, energy)
    """
    if debug:
        print(f"[DEBUG] Checking log file: {log_file_path}")
        print(f"[DEBUG]   File exists: {os.path.exists(log_file_path)}")
    
    if not os.path.exists(log_file_path):
        if debug:
            print(f"[DEBUG]   Log file does not exist, returning None")
        return None, None, None, None
    
    if debug:
        print(f"[DEBUG]   Reading log file...")
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Search for the "after forward pass" section with the dictionary
    pattern = r"after forward pass\s+edp: [\d.e+-]+, sub expressions: "
    match = re.search(pattern, content, re.DOTALL)
    
    if debug:
        if match:
            print(f"[DEBUG]   Found 'after forward pass' pattern at position {match.start()}")
        else:
            print(f"[DEBUG]   Pattern not found")
            # Try to find what's actually there
            if "after forward pass" in content:
                idx = content.find("after forward pass")
                snippet = content[idx:idx+500]
                print(f"[DEBUG]   Found 'after forward pass' at position {idx}, snippet:")
                print(f"[DEBUG]   {repr(snippet)}")
    
    if not match:
        if debug:
            print(f"[DEBUG]   No match found, returning None")
        return None, None, None, None
    
    # Extract the dictionary by finding the matching brace
    start_pos = match.end()
    brace_count = 0
    i = start_pos
    dict_start = None
    
    # Find the opening brace
    while i < len(content):
        if content[i] == '{':
            dict_start = i
            brace_count = 1
            i += 1
            break
        i += 1
    
    if dict_start is None:
        if debug:
            print(f"[DEBUG]   Could not find opening brace after 'sub expressions:'")
        return None, None, None, None
    
    # Find the matching closing brace
    while i < len(content) and brace_count > 0:
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count != 0:
        if debug:
            print(f"[DEBUG]   Could not find matching closing brace (unclosed braces: {brace_count})")
        return None, None, None, None
    
    # Extract the dictionary string
    sub_expr_str = content[dict_start:i]
    
    if debug:
        print(f"[DEBUG]   Using last match")
        print(f"[DEBUG]   Sub expressions string (first 200 chars): {sub_expr_str[:200]}...")
    
    try:
        # Replace 'inf' with '"inf"' to make it JSON parseable
        sub_expr_str = sub_expr_str.replace('inf', '"inf"')
        # Replace single quotes with double quotes for JSON
        sub_expr_str = sub_expr_str.replace("'", '"')
        
        if debug:
            print(f"[DEBUG]   Attempting to parse JSON (first 300 chars): {sub_expr_str[:300]}...")
        
        # Parse the dictionary string
        sub_expr_dict = json.loads(sub_expr_str)
        execution_time = sub_expr_dict.get('execution_time', None)
        clk_period = sub_expr_dict.get('clk_period', None)
        energy = sub_expr_dict.get('total energy', None)
        
        if debug:
            print(f"[DEBUG]   Parsed values: execution_time={execution_time}, clk_period={clk_period}, energy={energy}")
        
        if execution_time is not None and clk_period is not None:
            delay_cycles = execution_time / clk_period
            if debug:
                print(f"[DEBUG]   Calculated delay_cycles: {delay_cycles}")
            return execution_time, clk_period, delay_cycles, energy
        else:
            if debug:
                print(f"[DEBUG]   Missing required keys: execution_time={execution_time}, clk_period={clk_period}")
    except json.JSONDecodeError as e:
        print(f"Error parsing sub expressions dictionary as JSON: {e}")
        if debug:
            print(f"[DEBUG]   JSON decode error at position {e.pos}: {e.msg}")
            print(f"[DEBUG]   Problematic section: {sub_expr_str[max(0, e.pos-50):e.pos+50]}")
    except Exception as e:
        print(f"Error parsing sub expressions dictionary: {e}")
        if debug:
            print(f"[DEBUG]   Exception details: {type(e).__name__}: {str(e)}")
    
    if debug:
        print(f"[DEBUG]   Failed to extract delay, returning None")
    return None, None, None, None


# New helper: parse benchmark base name and workload tag (e.g., N16)
def _parse_benchmark_base_and_workload(benchmark_dir_name: str, parent_dir_name: str):
    # parent_dir_name: e.g., "forwardpass_sweep_N16"
    wl_match_parent = re.search(r'_N(\d+)', parent_dir_name or '')
    wl_match_bench = re.search(r'_N(\d+)', benchmark_dir_name or '')
    wl = wl_match_parent.group(1) if wl_match_parent else (wl_match_bench.group(1) if wl_match_bench else None)
    workload_tag = f"N{wl}" if wl else "N?"

    # Strip prefixes/suffixes to get a stable base (e.g., "2mm", "3mm", "atax")
    base = benchmark_dir_name
    base = re.sub(r'^benchmark_', '', base)
    base = re.sub(r'_sweep_test(_N\d+)?$', '', base)
    return base, workload_tag

def _find_mlir_file(benchmark_path: str, benchmark_name: str, base_name: str, debug: bool = False) -> str | None:
    # Common patterns first
    candidates = [
        os.path.join(benchmark_path, 'tmp', f'tmp_{benchmark_name}_delay_0', 'benchmark_setup', f'{benchmark_name}.mlir'),
        os.path.join(benchmark_path, 'tmp', f'tmp_{benchmark_name}_delay_0', f'{benchmark_name}.mlir'),
        # Observed in your tree: tmp/tmp_kernel_*_edp_0
        os.path.join(benchmark_path, 'tmp', f'tmp_kernel_{base_name}_edp_0', f'{benchmark_name}.mlir'),
        os.path.join(benchmark_path, 'tmp', f'tmp_{base_name}_edp_0', f'{benchmark_name}.mlir'),
    ]
    for p in candidates:
        if os.path.exists(p):
            if debug:
                print(f"[DEBUG]   MLIR candidate exists: {p}")
            return p

    # Fallback: search any .mlir under tmp
    tmp_dir = os.path.join(benchmark_path, 'tmp')
    if os.path.isdir(tmp_dir):
        for root, dirs, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith('.mlir'):
                    if debug:
                        print(f"[DEBUG]   Found MLIR by search: {os.path.join(root, f)}")
                    return os.path.join(root, f)
    return None

def _find_run_log(benchmark_path: str, debug: bool = False) -> str | None:
    # Prefer run_codesign.log anywhere under benchmark_path
    for root, dirs, files in os.walk(benchmark_path):
        for f in files:
            if f == 'run_codesign.log':
                p = os.path.join(root, f)
                if debug:
                    print(f"[DEBUG]   Found run_codesign.log: {p}")
                return p
    # Fallback: latest file in log/<timestamp> if present
    log_dir = os.path.join(benchmark_path, 'log')
    if os.path.isdir(log_dir):
        subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if subdirs:
            latest = max(subdirs, key=os.path.getmtime)
            # Try any *.log inside
            for root, dirs, files in os.walk(latest):
                for f in files:
                    if f.endswith('.log'):
                        p = os.path.join(root, f)
                        if debug:
                            print(f"[DEBUG]   Using fallback log file: {p}")
                        return p
    return None

def extract_benchmark_results(result_dir_path: str, debug: bool = False) -> dict:
    """
    Extract benchmark results from a result directory.
    """
    results = {}

    if debug:
        print(f"[DEBUG] Starting extract_benchmark_results")
        print(f"[DEBUG] Input path: {result_dir_path}")
    
    # Normalize path (remove trailing slash)
    original_path = result_dir_path
    result_dir_path = result_dir_path.rstrip('/')
    if debug and original_path != result_dir_path:
        print(f"[DEBUG] Normalized path (removed trailing slash): {result_dir_path}")
    
    scan_roots = []
    if result_dir_path.endswith('.list'):
        base_path = result_dir_path
        if not os.path.isdir(base_path):
            print(f"Error: Directory not found: {base_path}")
            return results

        # Collect all immediate subdirs that contain benchmark_* directories
        contents = os.listdir(base_path)
        if debug:
            print(f"[DEBUG] .list contents: {contents}")
        for item in contents:
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                sub_contents = os.listdir(item_path)
                has_benchmarks = any(d.startswith('benchmark') and os.path.isdir(os.path.join(item_path, d)) for d in sub_contents)
                if has_benchmarks:
                    scan_roots.append(item_path)
                    if debug:
                        print(f"[DEBUG]   Workload root added: {item_path}")
        if not scan_roots:
            # Fallback to common subdir names
            possible_subdirs = [os.path.basename(result_dir_path).replace('.list', ''), 'table_exp', 'forwardpass_sweep']
            if debug:
                print(f"[DEBUG] No workload subdirs found, trying fallback: {possible_subdirs}")
            for subdir in possible_subdirs:
                test_path = os.path.join(base_path, subdir)
                if os.path.isdir(test_path):
                    sub_contents = os.listdir(test_path)
                    if any(d.startswith('benchmark') and os.path.isdir(os.path.join(test_path, d)) for d in sub_contents):
                        scan_roots.append(test_path)
                        if debug:
                            print(f"[DEBUG]   Fallback workload root: {test_path}")
            if not scan_roots:
                print(f"Warning: Could not find expected subdirectory with benchmarks in {base_path}")
                print(f"Tried: {possible_subdirs}")
                print(f"Available directories: {[d for d in contents if os.path.isdir(os.path.join(base_path, d))]}")
    else:
        scan_roots.append(result_dir_path)

    if debug:
        print(f"[DEBUG] Final workload roots to scan: {scan_roots}")

    # Verify and process
    scan_roots = [r for r in scan_roots if os.path.isdir(r)]
    if not scan_roots:
        print(f"Error: No benchmark directories found under: {result_dir_path}")
        return results

    for root in scan_roots:
        if debug:
            print(f"[DEBUG] Scanning root: {root} contents={os.listdir(root)}")
        parent_dir_name = os.path.basename(root)

        for benchmark_name in os.listdir(root):
            benchmark_path = os.path.join(root, benchmark_name)
            if not os.path.isdir(benchmark_path) or not benchmark_name.startswith('benchmark'):
                continue

            base_name, workload_tag = _parse_benchmark_base_and_workload(benchmark_name, parent_dir_name)

            # MLIR discovery
            mlir_file = _find_mlir_file(benchmark_path, benchmark_name, base_name, debug=debug)

            # Log discovery
            log_file = _find_run_log(benchmark_path, debug=debug)

            num_ops = 0
            execution_time = 0.0
            delay_cycles = 0.0
            energy_value = 0.0

            if mlir_file and os.path.exists(mlir_file):
                if debug:
                    print(f"[DEBUG] Reading MLIR: {mlir_file}")
                with open(mlir_file, 'r') as f:
                    mlir_text = f.read()
                num_ops = count_ops(mlir_text)
            else:
                print(f"Warning: MLIR file not found for {benchmark_name}")

            if log_file:
                exec_time, clk_period, del_cycles, energy = extract_delay_from_log(log_file, debug=debug)
                if exec_time is not None:
                    execution_time = exec_time
                    delay_cycles = del_cycles
                else:
                    print(f"Warning: Could not extract delay from log for {benchmark_name}")
                energy_value = energy if energy is not None else 0.0
            else:
                if debug:
                    print(f"[DEBUG] No log file found under {benchmark_path}; energy/delay left at 0")

            metrics = {
                "NumberOfMLIROps": {"value": num_ops, "unit": "ops"},
                "Energy": {"value": energy_value, "unit": "nJ"},
                "delayCycles": {"value": delay_cycles, "unit": "cycles"},
                "executionTime": {"value": execution_time, "unit": "ns"},
            }

            if base_name not in results:
                results[base_name] = {}
            results[base_name][workload_tag] = metrics

            print(f"Processed {base_name} [{workload_tag}]: {num_ops} ops, {delay_cycles:.2f} cycles, {energy_value:.2f} nJ, {execution_time:.2f} ns")

    return results

def print_results_table(benchmark_data: dict):
    """
    Print a table. Supports nested dict: {base: {workload: metrics}}
    """
    if not benchmark_data:
        print("No benchmark data to display")
        return

    headers = ["Benchmark", "Workload", "MLIR Ops", "Energy (nJ)", "Delay Cycles", "Exec Time (ns)"]
    col_widths = [len(h) for h in headers]

    rows = []
    for base, variants in benchmark_data.items():
        if isinstance(variants, dict):
            for wl, m in variants.items():
                rows.append((base, wl, m))
        else:
            rows.append((base, "", variants))

    for base, wl, m in rows:
        col_widths[0] = max(col_widths[0], len(base))
        col_widths[1] = max(col_widths[1], len(wl))
        col_widths[2] = max(col_widths[2], len(f"{m['NumberOfMLIROps']['value']:,}"))
        col_widths[3] = max(col_widths[3], len(f"{m['Energy']['value']:.2f}"))
        col_widths[4] = max(col_widths[4], len(f"{m['delayCycles']['value']:.2f}"))
        col_widths[5] = max(col_widths[5], len(f"{m['executionTime']['value']:.2f}"))

    col_widths = [w + 2 for w in col_widths]
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-" * len(header_row)

    print("\n" + "=" * len(header_row))
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * len(header_row))
    print(header_row)
    print(separator)

    for base, wl, m in sorted(rows, key=lambda r: (r[0], r[1])):
        row = [
            base.ljust(col_widths[0]),
            wl.ljust(col_widths[1]),
            f"{m['NumberOfMLIROps']['value']:,}".rjust(col_widths[2]),
            f"{m['Energy']['value']:.2f}".rjust(col_widths[3]),
            f"{m['delayCycles']['value']:.2f}".rjust(col_widths[4]),
            f"{m['executionTime']['value']:.2f}".rjust(col_widths[5]),
        ]
        print(" | ".join(row))

    print(separator)

    # Summary
    if len(rows) > 1:
        total_ops = sum(m['NumberOfMLIROps']['value'] for _, _, m in rows)
        total_energy = sum(m['Energy']['value'] for _, _, m in rows)
        avg_delay = sum(m['delayCycles']['value'] for _, _, m in rows) / len(rows)
        avg_exec_time = sum(m['executionTime']['value'] for _, _, m in rows) / len(rows)
        print(f"\nSummary Statistics:")
        print(f"  Total MLIR Operations: {total_ops:,}")
        print(f"  Total Energy: {total_energy:.2f} nJ")
        print(f"  Average Delay Cycles: {avg_delay:.2f}")
        print(f"  Average Execution Time: {avg_exec_time:.2f} ns")
    print("=" * len(header_row) + "\n")

def save_results_to_csv(benchmark_data: dict, output_dir: str, debug: bool = False):
    """
    Save nested results with explicit Workload column.
    """
    if not benchmark_data:
        if debug:
            print(f"[DEBUG] No benchmark data to save to CSV")
        return

    csv_filename = "benchmark_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    if debug:
        print(f"[DEBUG] Saving CSV to: {csv_path}")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Benchmark',
            'Workload',
            'NumberOfMLIROps (ops)',
            'Energy (nJ)',
            'delayCycles (cycles)',
            'executionTime (ns)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for base, variants in benchmark_data.items():
            if isinstance(variants, dict):
                for wl, m in variants.items():
                    writer.writerow({
                        'Benchmark': base,
                        'Workload': wl,
                        'NumberOfMLIROps (ops)': m['NumberOfMLIROps']['value'],
                        'Energy (nJ)': m['Energy']['value'],
                        'delayCycles (cycles)': m['delayCycles']['value'],
                        'executionTime (ns)': m['executionTime']['value']
                    })
            else:
                m = variants
                writer.writerow({
                    'Benchmark': base,
                    'Workload': '',
                    'NumberOfMLIROps (ops)': m['NumberOfMLIROps']['value'],
                    'Energy (nJ)': m['Energy']['value'],
                    'delayCycles (cycles)': m['delayCycles']['value'],
                    'executionTime (ns)': m['executionTime']['value']
                })

    print(f"Results saved to CSV: {csv_path}")

def save_results_to_latex(benchmark_data: dict, output_dir: str, debug: bool = False):
    """
    Save results as LaTeX table grouped by kernel size (workload).
    Layout: rows=benchmarks, columns=workload sizes
    Matches the provided image format with bold headers and proper spacing.
    """
    if not benchmark_data:
        if debug:
            print(f"[DEBUG] No benchmark data to save to LaTeX")
        return

    # Collect all unique workloads and sort them numerically
    all_workloads = set()
    for base, variants in benchmark_data.items():
        if isinstance(variants, dict):
            all_workloads.update(variants.keys())
    
    workloads = sorted(all_workloads, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
    benchmarks = sorted(benchmark_data.keys())
    
    if debug:
        print(f"[DEBUG] Workloads: {workloads}, Benchmarks: {benchmarks}")
    
    # Extract numeric workload sizes for cleaner header (N4 -> 4, N16 -> 16)
    workload_nums = [re.search(r'\d+', wl).group() for wl in workloads]
    
    # Build LaTeX document
    tex_lines = []
    tex_lines.append(r"\documentclass{article}")
    tex_lines.append(r"\usepackage{booktabs}")
    tex_lines.append(r"\usepackage{amsmath}")
    tex_lines.append(r"\usepackage{array}")
    tex_lines.append(r"\begin{document}")
    tex_lines.append("")
    
    # ===== INDIVIDUAL METRIC TABLES =====
    metrics = [
        ("NumberOfMLIROps", "MLIR Operations", "ops"),
        ("Energy", "Energy", "nJ"),
        ("delayCycles", "Delay Cycles", "cycles"),
        ("executionTime", "Execution Time", "ns"),
    ]
    
    for metric_key, metric_label, metric_unit in metrics:
        tex_lines.append(f"% {metric_label} ({metric_unit})")
        tex_lines.append(r"\begin{table}[h!]")
        tex_lines.append(r"\centering")
        tex_lines.append(f"\\caption{{{metric_label} by Benchmark and Kernel Size}}")
        
        # Column spec: l for benchmark name, r for each workload
        col_spec = "l" + "r" * len(workloads)
        tex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        tex_lines.append(r"\toprule")
        
        # Header row with simplified workload sizes
        header = r"\textbf{Kernel Size}"
        for wl_num in workload_nums:
            header += f" & \\textbf{{{wl_num}}}"
        header += r" \\"
        tex_lines.append(header)
        tex_lines.append(r"\midrule")
        
        # Data rows
        for bench in benchmarks:
            row = f"{bench}"
            for wl in workloads:
                if bench in benchmark_data and isinstance(benchmark_data[bench], dict):
                    if wl in benchmark_data[bench]:
                        value = benchmark_data[bench][wl][metric_key]['value']
                        if isinstance(value, float):
                            if value == 0.0:
                                row += " & —"
                            else:
                                row += f" & {value:,.2f}"
                        else:
                            row += f" & {value:,}"
                    else:
                        row += " & —"
                else:
                    row += " & —"
            row += r" \\"
            tex_lines.append(row)
        
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append(r"\end{table}")
        tex_lines.append("")
    
    # ===== COMBINED TABLE: All metrics in one table =====
    tex_lines.append(r"\newpage")
    tex_lines.append(r"% Combined Table: All Metrics")
    tex_lines.append(r"\begin{table}[h!]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Combined Results: MLIR Ops, Energy, Delay Cycles, and Execution Time}")
    
    # For combined table: benchmark | 4(...) | 16(...) | etc.
    col_spec = "l" + "|c" * len(workloads)
    tex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    tex_lines.append(r"\toprule")
    
    # Main header row with simplified kernel sizes
    header = r"\textbf{Kernel Size}"
    for wl_num in workload_nums:
        header += f" & \\textbf{{{wl_num}}}"
    header += r" \\"
    tex_lines.append(header)
    
    # Sub-header row: ops, nJ, cycles, ns for each workload
    subheader = ""
    for _ in workload_nums:
        subheader += r" & \small{ops / nJ / cycles / ns}"
    subheader += r" \\"
    tex_lines.append(subheader)
    tex_lines.append(r"\midrule")
    
    # Data rows
    for bench in benchmarks:
        row = bench
        for wl in workloads:
            if bench in benchmark_data and isinstance(benchmark_data[bench], dict) and wl in benchmark_data[bench]:
                m = benchmark_data[bench][wl]
                ops = m['NumberOfMLIROps']['value']
                energy = m['Energy']['value']
                delay = m['delayCycles']['value']
                exec_time = m['executionTime']['value']
                
                # Format: ops / energy / delay / time
                if energy == 0.0 and delay == 0.0 and exec_time == 0.0:
                    row += " & —"
                else:
                    ops_str = f"{ops:,}" if isinstance(ops, int) else f"{ops:.0f}"
                    energy_str = f"{energy:,.2f}" if energy != 0.0 else "—"
                    delay_str = f"{delay:,.2f}" if delay != 0.0 else "—"
                    time_str = f"{exec_time:,.2f}" if exec_time != 0.0 else "—"
                    row += f" & {ops_str} / {energy_str} / {delay_str} / {time_str}"
            else:
                row += " & —"
        row += r" \\"
        tex_lines.append(row)
    
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")
    tex_lines.append("")
    
    tex_lines.append(r"\end{document}")
    
    # Write to file
    tex_filename = "benchmark_results.tex"
    tex_path = os.path.join(output_dir, tex_filename)
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    
    print(f"Results saved to LaTeX: {tex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract benchmark results from experiment result directory")
    parser.add_argument("result_path", type=str, help="Path to the experiment result directory (e.g., test/regression_results/forwardpass_sweep.list)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path (optional)")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print the final dictionary")
    parser.add_argument("--debug", "-d", action="store_true", help="Print detailed debugging information")
    
    args = parser.parse_args()
    
    # Determine the input directory (where CSV will be saved)
    input_dir = args.result_path.rstrip('/')
    if input_dir.endswith('.list'):
        csv_output_dir = input_dir
    else:
        csv_output_dir = os.path.dirname(input_dir) if os.path.isfile(input_dir) else input_dir
    
    benchmark_data = extract_benchmark_results(args.result_path, debug=args.debug)
    
    # Print human-readable table
    print_results_table(benchmark_data)
    
    # Print JSON results
    print("\n" + "="*80)
    print("Final Dictionary with All Stats (JSON):")
    print("="*80)
    
    if args.pretty:
        print(json.dumps(benchmark_data, indent=2))
    else:
        print(json.dumps(benchmark_data))
    
    # Save to CSV in the input directory
    save_results_to_csv(benchmark_data, csv_output_dir, debug=args.debug)
    
    # Save to LaTeX in the input directory
    save_results_to_latex(benchmark_data, csv_output_dir, debug=args.debug)
    
    # Save to JSON file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"Results saved to JSON: {args.output}")
    
    print(f"\nTotal benchmarks processed: {len(benchmark_data)}")
