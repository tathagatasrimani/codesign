#!/usr/bin/env python3
"""
Run Symbolic DESTINY Analysis

Main script that:
1. Runs C++ DESTINY or reads existing output
2. Creates symbolic model with SymPy expressions
3. Displays analysis
4. Exports symbolic model for framework integration
"""

import sys
import os
from symbolic_expressions import create_symbolic_model_from_destiny_output, MemoryAccessTimeExpressions
from symbolic_analysis_accurate import AccurateSymbolicAnalyzer
from parse_cpp_output import parse_cpp_destiny_output


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_symbolic_destiny.py <cpp_destiny_output> [--export <output.json>]")
        print("\nExample:")
        print("  python run_symbolic_destiny.py ../destiny_3d_cache-master/cpp_output_sram2layer.txt")
        print("  python run_symbolic_destiny.py ../destiny_3d_cache-master/cpp_output_sram2layer.txt --export model.json")
        sys.exit(1)

    cpp_output_file = sys.argv[1]
    export_file = None

    # Check for --export flag
    if len(sys.argv) >= 4 and sys.argv[2] == '--export':
        export_file = sys.argv[3]

    print("="*80)
    print("SYMBOLIC DESTINY ANALYSIS")
    print("="*80)

    # Step 1: Create symbolic model from C++ DESTINY output
    print(f"\n📁 Reading C++ DESTINY output: {cpp_output_file}")
    model = create_symbolic_model_from_destiny_output(cpp_output_file)

    print(f"\n✓ Created symbolic model:")
    print(f"  {model}")

    # Step 2: Display analysis
    print("\n" + "="*80)
    print("SYMBOLIC EXPRESSIONS FOR THIS MEMORY BLOCK")
    print("="*80)

    print("\n1️⃣  Bitline Delay Expression:")
    print(f"   τ_bitline = {model.expressions.tau_bitline}")
    print(f"\n   Expanded:")
    print(f"   τ_bitline = {model.expressions.tau_bitline_expanded}")

    print("\n2️⃣  Sense Amplifier Delay Expression:")
    print(f"   t_senseamp = {model.expressions.t_senseamp}")

    print("\n3️⃣  Decoder Stage Delay Expression:")
    print(f"   t_decoder_stage = {model.expressions.t_decoder_stage}")

    print("\n4️⃣  Mux Delay Expression:")
    print(f"   t_mux_level = {model.expressions.t_mux_level}")

    # Step 3: Show numerical results
    print("\n" + "="*80)
    print("NUMERICAL RESULTS (from C++ DESTINY)")
    print("="*80)

    print(f"\n  Component          Value            Expression Available")
    print(f"  {'─'*60}")
    for comp, value in model.numerical_results.items():
        if 'total' in comp:
            units = 'ns'
            val = value * 1e9
        elif 'bitline' in comp or 'decoder' in comp:
            units = 'ns'
            val = value * 1e9
        else:
            units = 'ps'
            val = value * 1e12

        has_expr = "✓" if model.get_symbolic_expression(comp.replace('t_', 'tau_bitline' if 'bitline' in comp else comp)) else "–"
        print(f"  {comp:<18} {val:8.3f} {units:<8} {has_expr}")

    # Step 4: Show configuration
    print("\n" + "="*80)
    print("CONFIGURATION PARAMETERS")
    print("="*80)
    print(f"\n  Subarray: {model.config_params['rows']} rows × {model.config_params['cols']} cols")
    print(f"  Banks: {model.config_params['num_banks_x']} × {model.config_params['num_banks_y']} × {model.config_params['num_stacks']}")
    print(f"  Mux: SA={model.config_params['senseamp_mux']}, L1={model.config_params['output_mux_l1']}, L2={model.config_params['output_mux_l2']}")

    # Step 5: Export if requested
    if export_file:
        print("\n" + "="*80)
        print("EXPORTING SYMBOLIC MODEL")
        print("="*80)
        model.export_to_json(export_file)
        print(f"\n✓ Model exported to: {export_file}")
        print("\nThe JSON file contains:")
        print("  • Symbolic expressions (SymPy format)")
        print("  • LaTeX representations")
        print("  • Numerical results from C++ DESTINY")
        print("  • Configuration parameters")
        print("\nThis can be used by other framework components!")

    # Step 6: Show how to use in framework
    print("\n" + "="*80)
    print("FRAMEWORK INTEGRATION EXAMPLE")
    print("="*80)

    print("""
from symbolic_expressions import create_symbolic_model_from_destiny_output

# Create model from DESTINY output
model = create_symbolic_model_from_destiny_output('cpp_output.txt')

# Get symbolic expression
bitline_expr = model.get_symbolic_expression('tau_bitline')
print(f"Bitline expression: {bitline_expr}")

# Get numerical value
bitline_value = model.get_numerical_value('t_bitline')
print(f"Bitline delay: {bitline_value*1e9:.3f} ns")

# Get all expressions
all_expressions = model.expressions.get_all_expressions()

# Evaluate with custom parameters
params = {'rows': 512, 'R_per_cell': 1e3, 'C_per_cell': 1e-15, ...}
result = model.expressions.evaluate('C_bitline', params)

# Export to JSON for other tools
model.export_to_json('memory_model.json')

# Convert to Python function
bitline_func = model.expressions.to_python_function('tau_bitline')
result = bitline_func(R_access=1e3, R_pulldown=500, ...)
""")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)

    # Also run the full analysis
    if '--full' in sys.argv:
        print("\n\nRunning full symbolic analysis...\n")
        cpp_config = parse_cpp_destiny_output(cpp_output_file)
        analyzer = AccurateSymbolicAnalyzer(cpp_config)
        analyzer.run_complete_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
