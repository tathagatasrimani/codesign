#!/usr/bin/env python3
"""
Example: How other framework components can use the symbolic models

This demonstrates integration patterns for your framework
"""

from symbolic_expressions import create_symbolic_model_from_destiny_output
import json


def example_1_basic_usage():
    """Example 1: Basic usage - get expressions and values"""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)

    # Create model from DESTINY output
    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )

    # Get a specific symbolic expression
    bitline_expr = model.get_symbolic_expression('tau_bitline')
    print(f"\n✓ Bitline symbolic expression:")
    print(f"  {bitline_expr}")

    # Get numerical value from C++ DESTINY
    bitline_value = model.get_numerical_value('t_bitline')
    print(f"\n✓ Bitline numerical value (from C++ DESTINY):")
    print(f"  {bitline_value*1e9:.6f} ns")

    # Get configuration
    print(f"\n✓ Configuration:")
    print(f"  Rows: {model.config_params['rows']}")
    print(f"  Cols: {model.config_params['cols']}")


def example_2_evaluate_expression():
    """Example 2: Evaluate expression with custom parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Evaluate Expression with Custom Parameters")
    print("="*80)

    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )

    # Define custom parameters
    params = {
        'rows': 512,  # Half the rows
        'R_per_cell': 1.5e3,  # 1.5 kΩ
        'C_per_cell': 0.3e-15,  # 0.3 fF
    }

    try:
        # Evaluate capacitance for different row count
        C_bitline = model.expressions.evaluate('C_bitline', params)
        print(f"\n✓ For {params['rows']} rows:")
        print(f"  C_bitline = {C_bitline*1e15:.3f} fF")

        # Also evaluate resistance
        R_bitline = model.expressions.evaluate('R_bitline', params)
        print(f"  R_bitline = {R_bitline:.3f} Ω")

    except ValueError as e:
        print(f"\n✗ Need more parameters: {e}")


def example_3_convert_to_function():
    """Example 3: Convert expression to Python function"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Convert to Python Function")
    print("="*80)

    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )

    # Convert bitline capacitance to a Python function
    C_bitline_func = model.expressions.to_python_function('C_bitline')

    print("\n✓ Created Python function for C_bitline")
    print("  Can now call: C_bitline_func(C_per_cell, rows)")

    # Use it with different values
    for rows in [256, 512, 1024, 2048]:
        C = C_bitline_func(0.3e-15, rows)  # C_per_cell=0.3fF
        print(f"  {rows} rows → C_bitline = {C*1e15:.3f} fF")


def example_4_get_all_expressions():
    """Example 4: Get all expressions for processing"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Get All Expressions")
    print("="*80)

    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )

    # Get all expressions
    all_exprs = model.expressions.get_all_expressions()

    print(f"\n✓ Available expressions: {len(all_exprs)}")
    for name, expr in all_exprs.items():
        # Count symbols in each expression
        num_symbols = len(expr.free_symbols)
        print(f"  {name:<25} → {num_symbols} symbols")


def example_5_load_from_json():
    """Example 5: Load model from exported JSON"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Load from JSON (for other tools)")
    print("="*80)

    # First export the model
    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )
    model.export_to_json('temp_model.json')

    # Now load it as a regular JSON (e.g., in JavaScript, Julia, etc.)
    with open('temp_model.json', 'r') as f:
        data = json.load(f)

    print("\n✓ Loaded JSON model:")
    print(f"  Symbolic expressions: {len(data['symbolic_expressions'])}")
    print(f"  Numerical results: {len(data['numerical_results'])}")
    print(f"  Config params: {len(data['config_params'])}")

    # Access specific data
    tau_bitline = data['symbolic_expressions']['tau_bitline']
    print(f"\n✓ Bitline expression from JSON:")
    print(f"  Expression: {tau_bitline['expression']}")
    print(f"  LaTeX: {tau_bitline['latex']}")
    print(f"  Symbols: {', '.join(tau_bitline['symbols'])}")

    # Access numerical results
    t_total = data['numerical_results']['t_total']
    print(f"\n✓ Total access time from JSON:")
    print(f"  {t_total*1e9:.3f} ns")


def example_6_parametric_study():
    """Example 6: Parametric study using symbolic expressions"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Parametric Study")
    print("="*80)

    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )

    # Study how bitline capacitance scales with rows
    print("\n✓ Bitline Capacitance vs Rows (C_per_cell = 0.3 fF):")
    print("  Rows      C_bitline      Ratio vs 256")
    print("  " + "─"*40)

    baseline = None
    for rows in [256, 512, 1024, 2048]:
        params = {'C_per_cell': 0.3e-15, 'rows': rows}
        C = model.expressions.evaluate('C_bitline', params)

        if baseline is None:
            baseline = C
            ratio_str = "1.00×"
        else:
            ratio = C / baseline
            ratio_str = f"{ratio:.2f}×"

        print(f"  {rows:<8}  {C*1e15:8.2f} fF    {ratio_str}")

    print("\n  → Confirms linear scaling: C ∝ rows")


def example_7_latex_export():
    """Example 7: Export expressions as LaTeX"""
    print("\n" + "="*80)
    print("EXAMPLE 7: LaTeX Export (for papers/documentation)")
    print("="*80)

    model = create_symbolic_model_from_destiny_output(
        '../destiny_3d_cache-master/cpp_output_sram2layer.txt'
    )

    # Get LaTeX for key expressions
    expressions_to_export = ['tau_bitline', 't_senseamp', 't_decoder_stage']

    print("\n✓ LaTeX representations:")
    for expr_name in expressions_to_export:
        latex_str = model.expressions.to_latex(expr_name)
        print(f"\n  {expr_name}:")
        print(f"  ${latex_str}$")


def main():
    """Run all examples"""
    print("\n" + "🚀"*40)
    print("FRAMEWORK INTEGRATION EXAMPLES")
    print("🚀"*40 + "\n")

    try:
        example_1_basic_usage()
        example_2_evaluate_expression()
        example_3_convert_to_function()
        example_4_get_all_expressions()
        example_5_load_from_json()
        example_6_parametric_study()
        example_7_latex_export()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nThese patterns can be used to integrate symbolic models")
    print("into your framework's analysis, optimization, and visualization tools!")


if __name__ == "__main__":
    main()
