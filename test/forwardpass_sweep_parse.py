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
    # Format: "after forward pass\n edp: <number>, sub expressions: {<dictionary>}"
    # The dictionary contains 'execution_time' and 'clk_period'
    
    # Find the position of "after forward pass" followed by "sub expressions:"
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


def extract_benchmark_results(result_dir_path: str, debug: bool = False) -> dict:
    """
    Extract benchmark results from a result directory.
    
    Args:
        result_dir_path: Path to the experiment result directory (e.g., test/regression_results/forwardpass_sweep.list)
        debug: If True, print detailed debugging information
    
    Returns:
        Dictionary with benchmark names as keys and metrics as values
    """
    results = {}
    
    if debug:
        print(f"[DEBUG] Starting extract_benchmark_results")
        print(f"[DEBUG] Input path: {result_dir_path}")
    
    # Normalize path (remove trailing slash, handle .list)
    original_path = result_dir_path
    result_dir_path = result_dir_path.rstrip('/')
    if debug and original_path != result_dir_path:
        print(f"[DEBUG] Normalized path (removed trailing slash): {result_dir_path}")
    
    # If path ends with .list, try to find the appropriate subdirectory
    if result_dir_path.endswith('.list'):
        if debug:
            print(f"[DEBUG] Path ends with .list, searching for subdirectory")
        base_path = result_dir_path
        # Extract the base name (e.g., "forwardpass_sweep" from "forwardpass_sweep.list")
        base_name = os.path.basename(result_dir_path).replace('.list', '')
        
        if debug:
            print(f"[DEBUG] Base name extracted: {base_name}")
        
        # Try common subdirectory names
        possible_subdirs = [base_name, 'table_exp', 'forwardpass_sweep']
        
        if debug:
            print(f"[DEBUG] Trying subdirectories: {possible_subdirs}")
        
        found_subdir = None
        for subdir in possible_subdirs:
            test_path = os.path.join(base_path, subdir)
            if debug:
                print(f"[DEBUG]   Checking: {test_path}")
                print(f"[DEBUG]     Exists: {os.path.exists(test_path)}, IsDir: {os.path.isdir(test_path) if os.path.exists(test_path) else 'N/A'}")
            
            if os.path.exists(test_path) and os.path.isdir(test_path):
                # Check if this subdirectory contains benchmark directories
                sub_contents = os.listdir(test_path)
                if debug:
                    print(f"[DEBUG]     Contents: {sub_contents}")
                
                benchmark_dirs = [d for d in sub_contents if os.path.isdir(os.path.join(test_path, d)) and d.startswith('benchmark')]
                if debug:
                    print(f"[DEBUG]     Benchmark directories found: {benchmark_dirs}")
                
                if benchmark_dirs:
                    result_dir_path = test_path
                    found_subdir = subdir
                    if debug:
                        print(f"[DEBUG]   Found subdirectory: {subdir} -> {result_dir_path}")
                    break
        
        if found_subdir is None:
            if debug:
                print(f"[DEBUG] No subdirectory found in first pass, trying fallback search")
            # If no subdirectory found, check if the .list path itself contains benchmarks
            if os.path.exists(base_path) and os.path.isdir(base_path):
                # Try to find a subdirectory that contains benchmark directories
                contents = os.listdir(base_path)
                if debug:
                    print(f"[DEBUG]   Base path contents: {contents}")
                
                for item in contents:
                    item_path = os.path.join(base_path, item)
                    if debug:
                        print(f"[DEBUG]   Checking item: {item_path}")
                    
                    if os.path.isdir(item_path):
                        sub_contents = os.listdir(item_path)
                        if debug:
                            print(f"[DEBUG]     Item contents: {sub_contents}")
                        
                        benchmark_dirs = [d for d in sub_contents if os.path.isdir(os.path.join(item_path, d)) and d.startswith('benchmark')]
                        if debug:
                            print(f"[DEBUG]     Benchmark directories: {benchmark_dirs}")
                        
                        if benchmark_dirs:
                            result_dir_path = item_path
                            found_subdir = item
                            if debug:
                                print(f"[DEBUG]   Found subdirectory in fallback: {item} -> {result_dir_path}")
                            break
                
                if found_subdir is None:
                    print(f"Warning: Could not find expected subdirectory with benchmarks in {base_path}")
                    print(f"Tried: {possible_subdirs}")
                    print(f"Available directories: {[d for d in contents if os.path.isdir(os.path.join(base_path, d))]}")
    
    if debug:
        print(f"[DEBUG] Final result_dir_path: {result_dir_path}")
    
    # Check if directory exists
    if not os.path.exists(result_dir_path):
        print(f"Error: Directory not found: {result_dir_path}")
        return results
    
    if debug:
        print(f"[DEBUG] Directory exists, listing contents...")
        contents = os.listdir(result_dir_path)
        print(f"[DEBUG] Contents: {contents}")
    
    # Iterate through subdirectories (each is a benchmark)
    for benchmark_name in os.listdir(result_dir_path):
        benchmark_path = os.path.join(result_dir_path, benchmark_name)
        
        if debug:
            print(f"\n[DEBUG] Processing: {benchmark_name}")
            print(f"[DEBUG]   Benchmark path: {benchmark_path}")
            print(f"[DEBUG]   Is directory: {os.path.isdir(benchmark_path)}")
            print(f"[DEBUG]   Starts with 'benchmark': {benchmark_name.startswith('benchmark')}")
        
        # Skip if not a directory or doesn't look like a benchmark
        if not os.path.isdir(benchmark_path):
            if debug:
                print(f"[DEBUG]   Skipping (not a directory)")
            continue
        
        # Only process directories that start with "benchmark"
        if not benchmark_name.startswith('benchmark'):
            if debug:
                print(f"[DEBUG]   Skipping (doesn't start with 'benchmark')")
            continue
        
        # Try to find MLIR file - check multiple possible locations
        mlir_file = None
        possible_mlir_paths = [
            # Standard path
            os.path.join(benchmark_path, 'tmp', f'tmp_{benchmark_name}_delay_0', 'benchmark_setup', f'{benchmark_name}.mlir'),
            # Alternative path patterns
            os.path.join(benchmark_path, 'tmp', f'tmp_{benchmark_name}_delay_0', f'{benchmark_name}.mlir'),
            os.path.join(benchmark_path, 'tmp', 'tmp_kernel_*_delay_0', 'benchmark_setup', '*.mlir'),
        ]
        
        if debug:
            print(f"[DEBUG]   Trying MLIR file paths:")
            for i, path in enumerate(possible_mlir_paths[:2]):
                print(f"[DEBUG]     [{i+1}] {path}")
                print(f"[DEBUG]         Exists: {os.path.exists(path)}")
        
        # First try exact paths
        for path in possible_mlir_paths[:2]:  # First two are exact paths
            if os.path.exists(path):
                mlir_file = path
                if debug:
                    print(f"[DEBUG]   Found MLIR file at: {mlir_file}")
                break
        
        # If not found, search recursively in tmp directory
        if mlir_file is None:
            tmp_dir = os.path.join(benchmark_path, 'tmp')
            if debug:
                print(f"[DEBUG]   MLIR file not found in standard paths, searching recursively in: {tmp_dir}")
                print(f"[DEBUG]     tmp_dir exists: {os.path.exists(tmp_dir)}")
            
            if os.path.exists(tmp_dir):
                # First try to find MLIR file with benchmark name in it
                benchmark_base = benchmark_name.replace('benchmark_', '').replace('_sweep_test', '').replace('_test', '')
                if debug:
                    print(f"[DEBUG]     Extracted benchmark base: {benchmark_base}")
                    print(f"[DEBUG]     Searching for MLIR files matching benchmark name...")
                
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file.endswith('.mlir'):
                            if debug:
                                print(f"[DEBUG]       Found .mlir file: {os.path.join(root, file)}")
                            # Extract base name from benchmark (e.g., "atax" from "benchmark_atax_sweep_test")
                            if benchmark_base in file or benchmark_name in file:
                                mlir_file = os.path.join(root, file)
                                if debug:
                                    print(f"[DEBUG]     Matched MLIR file: {mlir_file}")
                                break
                    if mlir_file:
                        break
                
                # If still not found, just take the first .mlir file found
                if mlir_file is None:
                    if debug:
                        print(f"[DEBUG]     No match found, taking first .mlir file...")
                    for root, dirs, files in os.walk(tmp_dir):
                        for file in files:
                            if file.endswith('.mlir'):
                                mlir_file = os.path.join(root, file)
                                if debug:
                                    print(f"[DEBUG]     Using first .mlir file found: {mlir_file}")
                                break
                        if mlir_file:
                            break
        
        # Construct path to run_codesign.log
        log_file = os.path.join(benchmark_path, 'run_codesign.log')
        
        if debug:
            print(f"[DEBUG]   Log file path: {log_file}")
            print(f"[DEBUG]     Exists: {os.path.exists(log_file)}")
        
        num_ops = 0
        execution_time = 0
        delay_cycles = 0
        
        # Check if MLIR file exists and count ops
        if mlir_file and os.path.exists(mlir_file):
            if debug:
                print(f"[DEBUG]   Reading MLIR file: {mlir_file}")
            with open(mlir_file, 'r') as f:
                mlir_text = f.read()
            if debug:
                print(f"[DEBUG]     MLIR file size: {len(mlir_text)} characters")
            num_ops = count_ops(mlir_text)
            if debug:
                print(f"[DEBUG]     Counted {num_ops} MLIR operations")
        else:
            print(f"Warning: MLIR file not found for {benchmark_name}")
            if mlir_file:
                print(f"  Searched at: {mlir_file}")
            if debug:
                print(f"[DEBUG]   MLIR file is None or doesn't exist")
        
        # Extract delay and energy from log file
        exec_time, clk_period, del_cycles, energy = extract_delay_from_log(log_file, debug=debug)
        if exec_time is not None:
            execution_time = exec_time
            delay_cycles = del_cycles
            if debug:
                print(f"[DEBUG]   Extracted delay info: execution_time={execution_time}, delay_cycles={delay_cycles}, energy={energy}")
        else:
            print(f"Warning: Could not extract delay from log for {benchmark_name}")
            if debug:
                print(f"[DEBUG]   Failed to extract delay from log")
        
        # Use energy from log if available, otherwise default to 0
        energy_value = energy if energy is not None else 0
        
        results[benchmark_name] = {
            "NumberOfMLIROps": {
                "value": num_ops,
                "unit": "ops"
            },
            "Energy": {
                "value": energy_value,
                "unit": "nJ"  # nanojoules, based on codebase references to "Dynamic write energy (nJ)"
            },
            "delayCycles": {
                "value": delay_cycles,
                "unit": "cycles"
            },
            "executionTime": {
                "value": execution_time,
                "unit": "ns"  # nanoseconds, based on "clk_period": "Clock Period over generations (ns)"
            }
        }
        if debug:
            print(f"[DEBUG]   Final stats for {benchmark_name}: {results[benchmark_name]}")
        print(f"Processed {benchmark_name}: {num_ops} ops, {delay_cycles:.2f} cycles, {energy_value:.2f} nJ, {execution_time:.2f} ns")
    
    return results


def print_results_table(benchmark_data: dict):
    """
    Print a human-readable formatted table of benchmark results to the terminal.
    
    Args:
        benchmark_data: Dictionary with benchmark results
    """
    if not benchmark_data:
        print("No benchmark data to display")
        return
    
    # Define column headers and widths
    headers = ["Benchmark", "MLIR Ops", "Energy (nJ)", "Delay Cycles", "Exec Time (ns)"]
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for benchmark_name, metrics in benchmark_data.items():
        col_widths[0] = max(col_widths[0], len(benchmark_name))
        col_widths[1] = max(col_widths[1], len(f"{metrics['NumberOfMLIROps']['value']:,}"))
        col_widths[2] = max(col_widths[2], len(f"{metrics['Energy']['value']:.2f}"))
        col_widths[3] = max(col_widths[3], len(f"{metrics['delayCycles']['value']:.2f}"))
        col_widths[4] = max(col_widths[4], len(f"{metrics['executionTime']['value']:.2f}"))
    
    # Add padding
    col_widths = [w + 2 for w in col_widths]
    
    # Print table header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-" * len(header_row)
    
    print("\n" + "=" * len(header_row))
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * len(header_row))
    print(header_row)
    print(separator)
    
    # Print data rows
    for benchmark_name, metrics in sorted(benchmark_data.items()):
        row = [
            benchmark_name.ljust(col_widths[0]),
            f"{metrics['NumberOfMLIROps']['value']:,}".rjust(col_widths[1]),
            f"{metrics['Energy']['value']:.2f}".rjust(col_widths[2]),
            f"{metrics['delayCycles']['value']:.2f}".rjust(col_widths[3]),
            f"{metrics['executionTime']['value']:.2f}".rjust(col_widths[4])
        ]
        print(" | ".join(row))
    
    print(separator)
    
    # Print summary statistics
    if len(benchmark_data) > 1:
        total_ops = sum(m['NumberOfMLIROps']['value'] for m in benchmark_data.values())
        total_energy = sum(m['Energy']['value'] for m in benchmark_data.values())
        avg_delay = sum(m['delayCycles']['value'] for m in benchmark_data.values()) / len(benchmark_data)
        avg_exec_time = sum(m['executionTime']['value'] for m in benchmark_data.values()) / len(benchmark_data)
        
        print(f"\nSummary Statistics:")
        print(f"  Total MLIR Operations: {total_ops:,}")
        print(f"  Total Energy: {total_energy:.2f} nJ")
        print(f"  Average Delay Cycles: {avg_delay:.2f}")
        print(f"  Average Execution Time: {avg_exec_time:.2f} ns")
    
    print("=" * len(header_row) + "\n")


def save_results_to_csv(benchmark_data: dict, output_dir: str, debug: bool = False):
    """
    Save benchmark results to a CSV file in the output directory.
    
    Args:
        benchmark_data: Dictionary with benchmark results
        output_dir: Directory where the CSV file should be saved
        debug: If True, print debug information
    """
    if not benchmark_data:
        if debug:
            print(f"[DEBUG] No benchmark data to save to CSV")
        return
    
    # Determine the CSV filename
    csv_filename = "benchmark_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    if debug:
        print(f"[DEBUG] Saving CSV to: {csv_path}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Benchmark',
            'NumberOfMLIROps (ops)',
            'Energy (nJ)',
            'delayCycles (cycles)',
            'executionTime (ns)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for benchmark_name, metrics in benchmark_data.items():
            writer.writerow({
                'Benchmark': benchmark_name,
                'NumberOfMLIROps (ops)': metrics['NumberOfMLIROps']['value'],
                'Energy (nJ)': metrics['Energy']['value'],
                'delayCycles (cycles)': metrics['delayCycles']['value'],
                'executionTime (ns)': metrics['executionTime']['value']
            })
    
    print(f"Results saved to CSV: {csv_path}")


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
        # Use the .list directory as the output location
        csv_output_dir = input_dir
    else:
        # If not a .list directory, use the directory containing the path
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
    
    # Save to JSON file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"Results saved to JSON: {args.output}")
    
    print(f"\nTotal benchmarks processed: {len(benchmark_data)}")
