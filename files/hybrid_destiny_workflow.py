#!/usr/bin/env python3
"""
Hybrid C++ DESTINY + Python DESTINY Workflow
==============================================

This script integrates:
1. C++ DESTINY - Performs design space exploration (DSE) to find optimal configuration
2. Python DESTINY - Computes symbolic expressions for memory access time using optimal config

Workflow:
  Input Config → C++ DESTINY (DSE) → Optimal Config → Python DESTINY (Symbolic) → Expressions
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_cpp_destiny(config_file_name: str, output_file_name: str) -> int:
    """
    Run C++ DESTINY for design space exploration

    Args:
        config_file_name: Name of configuration file
        output_file_name: Name of output file
        cpp_destiny_path: Path to C++ DESTINY executable

    Returns:
        Return code (0 = success)
    """
    cpp_destiny_path = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache/destiny")
    config_folder = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache/config")
    config_file_path = os.path.join(config_folder, config_file_name)
    output_folder = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache/output")
    output_file_path = os.path.join(output_folder, output_file_name)
    # Convert to absolute path before changing directories
    output_file_path = os.path.abspath(output_file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    old_cwd = os.getcwd()
    print("=" * 80)
    print("STEP 1: Running C++ DESTINY for Design Space Exploration")
    print("=" * 80)
    print(f"\nConfiguration: {config_file_name}")
    print(f"Output file: {output_file_name}")
    print("\nThis may take several minutes as C++ DESTINY explores the design space...")
    print("(Finding optimal configuration for cache organization)\n")

    os.chdir(config_folder)

    # DESTINY executable doesn't support -o flag and only writes to file in full_exploration mode
    # For other modes, it prints to stdout, so we capture stdout and write to file
    # We use Popen to both display output in real-time and save it to file
    cmd = [cpp_destiny_path, config_file_path]

    try:
        print(f"Running command: {cmd} from {os.getcwd()}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Write output to file while also displaying it
        with open(output_file_path, 'w') as f:
            for line in process.stdout:
                print(line, end='')  # Display in real-time
                f.write(line)  # Save to file
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        print("\n✓ C++ DESTINY DSE completed successfully!")
        print(f"  Optimal configuration saved to: {output_file_path}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ C++ DESTINY failed with error code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n✗ C++ DESTINY executable not found: {cpp_destiny_path}")
        return 1
    finally:
        os.chdir(old_cwd)


def run_python_symbolic_analysis(cpp_output_file_name: str, config_file_name: str, python_script: str) -> int:
    """
    Run Python DESTINY for symbolic access time analysis

    Args:
        cpp_output_file_name: C++ DESTINY output file name with optimal configuration
        config_file_name: Original configuration file name
        python_script: Path to symbolic analysis Python script

    Returns:
        Return code (0 = success)
    """
    python_script_path = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache_python", python_script)
    config_folder = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache/config")
    config_file_path = os.path.join(config_folder, config_file_name)
    cpp_output_folder = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache/output")
    cpp_output_file_path = os.path.join(cpp_output_folder, cpp_output_file_name)
    if not os.path.exists(cpp_output_folder):
        os.makedirs(cpp_output_folder)
    old_cwd = os.getcwd()
    print("\n" + "=" * 80)
    print("STEP 2: Running Python DESTINY for Symbolic Analysis")
    print("=" * 80)
    print(f"\nUsing optimal configuration from: {cpp_output_file_path}")
    print("Computing symbolic expressions for memory access time...\n")

    os.chdir(os.path.dirname(python_script_path))

    cmd = ["python", python_script_path, cpp_output_file_path, config_file_path]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✓ Python DESTINY symbolic analysis completed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Python DESTINY failed with error code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n✗ Python script not found: {python_script_path}")
        return 1
    finally:
        os.chdir(old_cwd)

def main():
    """Main workflow orchestrator"""
    parser = argparse.ArgumentParser(
        description="Hybrid C++ DESTINY + Python DESTINY Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample SRAM configuration
  python hybrid_destiny_workflow.py -c config/sample_SRAM_2layer.cfg

  # Specify custom paths
  python hybrid_destiny_workflow.py -c my_config.cfg \\
      --cpp-destiny ../destiny_3d_cache-master/destiny \\
      --output results/my_output.txt

Notes:
  - C++ DESTINY performs design space exploration (may take several minutes)
  - Python DESTINY uses the optimal configuration for symbolic analysis
  - All symbolic expressions and sensitivity analysis are printed to console
        """
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Configuration file name (e.g., sample_SRAM_2layer.cfg)"
    )

    parser.add_argument(
        "--cpp-destiny",
        type=str,
        default="destiny_3d_cache-master/destiny",
        help="Path to C++ DESTINY executable (default: destiny_3d_cache-master/destiny)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="hybrid_output.txt",
        help="Output file for C++ DESTINY results (default: hybrid_output.txt)"
    )

    parser.add_argument(
        "--python-script",
        type=str,
        default="symbolic_access_time_FIXED.py",
        help="Name of Python symbolic analysis script"
    )

    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip C++ DESTINY and use existing output file"
    )

    args = parser.parse_args()

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Hybrid C++ DESTINY + Python DESTINY Workflow" + " " * 18 + "║")
    print("╚" + "=" * 78 + "╝")

    print("\nWorkflow:")
    print("  1. C++ DESTINY performs Design Space Exploration (DSE)")
    print("  2. Extracts optimal configuration from DSE results")
    print("  3. Python DESTINY computes symbolic expressions for access time")
    print("  4. Performs sensitivity analysis on symbolic expressions")

    # Step 1: Run C++ DESTINY (unless skipped)
    if not args.skip_cpp:
        ret = run_cpp_destiny(args.config, args.output)
        if ret != 0:
            print("\n✗ Workflow failed at C++ DESTINY stage")
            return ret
    else:
        print("\n⚠ Skipping C++ DESTINY (using existing output)")
        output_file_path = os.path.join(os.path.dirname(__file__), "../destiny_3d_cache/output", args.output)
        if not os.path.exists(output_file_path):
            print(f"✗ Output file not found: {output_file_path}")
            return 1

    # Step 2: Run Python DESTINY symbolic analysis
    ret = run_python_symbolic_analysis(args.output, args.config, args.python_script)
    if ret != 0:
        print("\n✗ Workflow failed at Python DESTINY stage")
        return ret

    # Success!
    print("\n" + "=" * 80)
    print("✓ HYBRID WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOutputs:")
    print(f"  • C++ DESTINY results: {args.output}")
    print(f"  • Symbolic expressions: Printed above")
    print("\nNext Steps:")
    print("  • Use symbolic expressions for design space analysis")
    print("  • Perform parameter sweeps with different technology nodes")
    print("  • Compute derivatives for sensitivity analysis")
    print("  • Generate optimization constraints")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
