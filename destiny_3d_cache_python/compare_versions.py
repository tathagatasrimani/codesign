#!/usr/bin/env python3
"""
Compare C++ and Python DESTINY across multiple configurations
"""

import subprocess
import re
import os

CPP_DIR = "/Users/aaravwattal/Documents/Stanford/Junior Fall/Memory Modeling RSG/cacti_destiny_old/destiny_3d_cache-master"
PYTHON_DIR = "/Users/aaravwattal/Documents/Stanford/Junior Fall/Memory Modeling RSG/cacti_destiny_old/destiny_3d_cache_python"

# Test configurations
configs = [
    "sample_SRAM_2layer",
    "sample_SRAM_4layer",
    "sample_STTRAM",
    "sample_PCRAM",
    "sample_2D_eDRAM"
]

def extract_metrics(output_text):
    """Extract Area, Latency, and Energy from output"""
    metrics = {}

    # Extract Area
    area_match = re.search(r'Area\s*=\s*([\d.]+)\s*mm', output_text)
    if area_match:
        metrics['area'] = float(area_match.group(1))

    # Extract Read Latency
    latency_match = re.search(r'Read Latency\s*=\s*([\d.]+)\s*ns', output_text)
    if latency_match:
        metrics['latency'] = float(latency_match.group(1))

    # Extract Read Dynamic Energy
    energy_match = re.search(r'-\s*Read Dynamic Energy\s*=\s*([\d.]+)\s*pJ', output_text)
    if energy_match:
        metrics['energy'] = float(energy_match.group(1))

    return metrics

def run_cpp_destiny(config):
    """Run C++ DESTINY and return output"""
    os.chdir(CPP_DIR)
    cmd = f"./destiny config/{config}.cfg"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.stdout + result.stderr
    except Exception as e:
        return f"ERROR: {e}"

def run_python_destiny(config):
    """Run Python DESTINY and return output"""
    os.chdir(PYTHON_DIR)
    cmd = f"python main.py -config config/{config}.cfg"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.stdout + result.stderr
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("=" * 80)
    print("DESTINY C++ vs Python Comparison")
    print("=" * 80)
    print()

    results = []

    for config in configs:
        print(f"Testing: {config}")
        print("-" * 80)

        # Run C++ version
        print("  Running C++ DESTINY...")
        cpp_output = run_cpp_destiny(config)
        cpp_metrics = extract_metrics(cpp_output)

        # Run Python version
        print("  Running Python DESTINY...")
        python_output = run_python_destiny(config)
        python_metrics = extract_metrics(python_output)

        # Calculate differences
        result = {
            'config': config,
            'cpp': cpp_metrics,
            'python': python_metrics
        }

        if cpp_metrics and python_metrics:
            if 'area' in cpp_metrics and 'area' in python_metrics:
                area_diff = abs(cpp_metrics['area'] - python_metrics['area']) / cpp_metrics['area'] * 100
                result['area_diff'] = area_diff

            if 'latency' in cpp_metrics and 'latency' in python_metrics:
                latency_diff = abs(cpp_metrics['latency'] - python_metrics['latency']) / cpp_metrics['latency'] * 100
                result['latency_diff'] = latency_diff

            if 'energy' in cpp_metrics and 'energy' in python_metrics:
                energy_diff = abs(cpp_metrics['energy'] - python_metrics['energy']) / cpp_metrics['energy'] * 100
                result['energy_diff'] = energy_diff

        results.append(result)
        print(f"  ✓ Completed\n")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    for result in results:
        config = result['config']
        print(f"\n{config}:")
        print("-" * 80)

        if result['cpp'] and result['python']:
            print(f"  {'Metric':<20} {'C++':<15} {'Python':<15} {'Diff %':<10}")
            print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

            if 'area' in result['cpp'] and 'area' in result['python']:
                print(f"  {'Area (mm²)':<20} {result['cpp']['area']:<15.3f} {result['python']['area']:<15.3f} {result.get('area_diff', 0):<10.2f}%")

            if 'latency' in result['cpp'] and 'latency' in result['python']:
                print(f"  {'Latency (ns)':<20} {result['cpp']['latency']:<15.3f} {result['python']['latency']:<15.3f} {result.get('latency_diff', 0):<10.2f}%")

            if 'energy' in result['cpp'] and 'energy' in result['python']:
                print(f"  {'Energy (pJ)':<20} {result['cpp']['energy']:<15.3f} {result['python']['energy']:<15.3f} {result.get('energy_diff', 0):<10.2f}%")
        else:
            print("  ERROR: Failed to extract metrics")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
