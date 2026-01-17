import os
import re
import argparse
import json

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


def extract_delay_from_log(log_file_path: str):
    """
    Extract delay and clk_period from run_codesign.log file.
    Returns a tuple (execution_time, clk_period, delay_cycles)
    """
    if not os.path.exists(log_file_path):
        return None, None, None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Search for the "after forward pass" section with the dictionary
    pattern = r"after forward pass\s+delay: ([\d.e+-]+), sub expressions: (\{.+\})"
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None, None, None
    
    # Get the last match (most recent forward pass)
    delay_value, sub_expr_str = matches[-1]
    
    try:
        # Replace 'inf' with '"inf"' to make it JSON parseable
        sub_expr_str = sub_expr_str.replace('inf', '"inf"')
        # Replace single quotes with double quotes for JSON
        sub_expr_str = sub_expr_str.replace("'", '"')
        
        # Parse the dictionary string
        sub_expr_dict = json.loads(sub_expr_str)
        execution_time = sub_expr_dict.get('execution_time', None)
        clk_period = sub_expr_dict.get('clk_period', None)
        
        if execution_time and clk_period:
            delay_cycles = execution_time / clk_period
            return execution_time, clk_period, delay_cycles
    except Exception as e:
        print(f"Error parsing sub expressions dictionary: {e}")
        pass
    
    return None, None, None


def extract_benchmark_results(result_dir_path: str) -> dict:
    """
    Extract benchmark results from a result directory.
    
    Args:
        result_dir_path: Path to the experiment result directory (e.g., test/regression_results/table_exp.list/table_exp)
    
    Returns:
        Dictionary with benchmark names as keys and metrics as values
    """
    results = {}
    
    # If path ends with .list, append /table_exp
    if result_dir_path.endswith('.list'):
        result_dir_path = os.path.join(result_dir_path, 'table_exp')
    
    # Check if directory exists
    if not os.path.exists(result_dir_path):
        print(f"Error: Directory not found: {result_dir_path}")
        return results
    
    # Iterate through subdirectories (each is a benchmark)
    for benchmark_name in os.listdir(result_dir_path):
        benchmark_path = os.path.join(result_dir_path, benchmark_name)
        
        # Skip if not a directory
        if not os.path.isdir(benchmark_path):
            continue
        
        # Construct path to MLIR file
        mlir_file = os.path.join(
            benchmark_path, 
            'tmp', 
            f'tmp_{benchmark_name}_delay_0', 
            'benchmark_setup', 
            f'{benchmark_name}.mlir'
        )
        
        # Construct path to run_codesign.log
        log_file = os.path.join(benchmark_path, 'run_codesign.log')
        
        num_ops = 0
        execution_time = 0
        delay_cycles = 0
        
        # Check if MLIR file exists and count ops
        if os.path.exists(mlir_file):
            with open(mlir_file, 'r') as f:
                mlir_text = f.read()
            num_ops = count_ops(mlir_text)
        else:
            print(f"Warning: MLIR file not found for {benchmark_name} at {mlir_file}")
        
        # Extract delay from log file
        exec_time, clk_period, del_cycles = extract_delay_from_log(log_file)
        if exec_time is not None:
            execution_time = exec_time
            delay_cycles = del_cycles
        else:
            print(f"Warning: Could not extract delay from log for {benchmark_name}")
        
        results[benchmark_name] = {
            "NumberOfMLIROps": num_ops,
            "Energy": 0,
            "delayCycles": delay_cycles,
            "executionTime": execution_time
        }
        print(f"Processed {benchmark_name}: {num_ops} ops, {delay_cycles:.2f} cycles")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract benchmark results from experiment result directory")
    parser.add_argument("result_path", type=str, help="Path to the experiment result directory (e.g., test/regression_results/table_exp.list)")
    
    args = parser.parse_args()
    
    benchmark_data = extract_benchmark_results(args.result_path)
    print("\nBenchmark Results:")
    for benchmark_name, metrics in benchmark_data.items():
        print(f"{benchmark_name}: {metrics}")
