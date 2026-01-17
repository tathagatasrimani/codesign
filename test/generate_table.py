import os
import re
import argparse

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
        
        # Check if MLIR file exists
        if os.path.exists(mlir_file):
            with open(mlir_file, 'r') as f:
                mlir_text = f.read()
            
            num_ops = count_ops(mlir_text)
            
            results[benchmark_name] = {
                "NumberOfMLIROps": num_ops,
                "Energy": 0,
                "delayCycles": 0
            }
            print(f"Processed {benchmark_name}: {num_ops} ops")
        else:
            print(f"Warning: MLIR file not found for {benchmark_name} at {mlir_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract benchmark results from experiment result file")
    parser.add_argument("result_file", type=str, help="Path to the experiment result file (e.g., test/regression_results/table_exp.list)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.result_file):
        benchmark_data = extract_benchmark_results(args.result_file)
        print("\nBenchmark Results:")
        for benchmark_name, metrics in benchmark_data.items():
            print(f"{benchmark_name}: {metrics}")
    else:
        print(f"Result file not found: {args.result_file}")
