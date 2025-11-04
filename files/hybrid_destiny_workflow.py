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


def run_cpp_destiny(config_file: str, output_file: str, cpp_destiny_path: str) -> int:
    """
    Run C++ DESTINY for design space exploration

    Args:
        config_file: Path to configuration file
        output_file: Path to save C++ DESTINY output
        cpp_destiny_path: Path to C++ DESTINY executable

    Returns:
        Return code (0 = success)
    """
    print("=" * 80)
    print("STEP 1: Running C++ DESTINY for Design Space Exploration")
    print("=" * 80)
    print(f"\nConfiguration: {config_file}")
    print(f"Output file: {output_file}")
    print("\nThis may take several minutes as C++ DESTINY explores the design space...")
    print("(Finding optimal configuration for cache organization)\n")

    cmd = [cpp_destiny_path, config_file, "-o", output_file]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✓ C++ DESTINY DSE completed successfully!")
        print(f"  Optimal configuration saved to: {output_file}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ C++ DESTINY failed with error code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\n✗ C++ DESTINY executable not found: {cpp_destiny_path}")
        return 1


def run_python_symbolic_analysis(cpp_output_file: str, config_file: str, python_script: str) -> int:
    """
    Run Python DESTINY for symbolic access time analysis

    Args:
        cpp_output_file: C++ DESTINY output file with optimal configuration
        config_file: Original configuration file
        python_script: Path to symbolic analysis Python script

    Returns:
        Return code (0 = success)
    """
    print("\n" + "=" * 80)
    print("STEP 2: Running Python DESTINY for Symbolic Analysis")
    print("=" * 80)
    print(f"\nUsing optimal configuration from: {cpp_output_file}")
    print("Computing symbolic expressions for memory access time...\n")

    cmd = ["python", python_script, cpp_output_file, config_file]

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
        print(f"\n✗ Python script not found: {python_script}")
        return 1


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
        help="Configuration file path (e.g., config/sample_SRAM_2layer.cfg)"
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
        default="destiny_3d_cache_python/symbolic_access_time.py",
        help="Path to Python symbolic analysis script"
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
        ret = run_cpp_destiny(args.config, args.output, args.cpp_destiny)
        if ret != 0:
            print("\n✗ Workflow failed at C++ DESTINY stage")
            return ret
    else:
        print("\n⚠ Skipping C++ DESTINY (using existing output)")
        if not os.path.exists(args.output):
            print(f"✗ Output file not found: {args.output}")
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
