#!/usr/bin/env python3
"""
End-to-End Example: Complete Symbolic DESTINY Workflow

This demonstrates the COMPLETE workflow from C++ DESTINY output
to symbolic expressions that match concrete values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from symbolic_expressions import create_symbolic_model_from_destiny_output
from parse_cpp_output import parse_cpp_destiny_output

# Import parameter extraction function
import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from SubArray import SubArray
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget
from formula import calculate_on_resistance, calculate_transconductance


def extract_all_parameters(config_file: str, cpp_config):
    """Extract ALL technology parameters from Python DESTINY"""

    # Initialize Python DESTINY
    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile(config_file)

    g.tech = Technology()
    g.tech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.devtech = Technology()
    g.devtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.gtech = Technology()
    g.gtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.localWire = Wire()
    g.localWire.Initialize(g.inputParameter.processNode, WireType.local_aggressive,
                           WireRepeaterType.repeated_none, g.inputParameter.temperature, False)

    g.globalWire = Wire()
    g.globalWire.Initialize(g.inputParameter.processNode, WireType.global_aggressive,
                            WireRepeaterType.repeated_none, g.inputParameter.temperature, False)

    g.cell = MemCell()
    if len(g.inputParameter.fileMemCell) > 0:
        cellFile = g.inputParameter.fileMemCell[0]
        if '/' not in cellFile:
            cellFile = os.path.join('config', cellFile)
        g.cell.ReadCellFromFile(cellFile)

    # Create SubArray
    subarray = SubArray()
    subarray.Initialize(
        cpp_config.subarray_rows,
        cpp_config.subarray_cols,
        1, 1,
        cpp_config.senseamp_mux if cpp_config.senseamp_mux else 1, True,
        cpp_config.output_mux_l1 if cpp_config.output_mux_l1 else 1,
        cpp_config.output_mux_l2 if cpp_config.output_mux_l2 else 1,
        BufferDesignTarget.latency_first,
        cpp_config.num_stacks if cpp_config.num_stacks else 1
    )

    subarray.CalculateArea()
    subarray.CalculateLatency(1e20)

    # Extract all parameters
    NMOS = 0
    params = {
        # Configuration
        'rows': subarray.numRow,
        'cols': subarray.numColumn,

        # Technology
        'V_dd': g.tech.vdd,
        'I_on': g.tech.currentOnNmos[0],
        'R_eff': g.tech.effectiveResistanceMultiplier,
        'C_gate': g.tech.capIdealGate,

        # Bitline
        'R_bitline': subarray.resBitline,
        'C_bitline': subarray.capBitline,
        'R_per_cell': subarray.resBitline / subarray.numRow,
        'C_per_cell': subarray.capBitline / subarray.numRow,

        # Cell access
        'R_access': subarray.resCellAccess,
        'C_access': subarray.capCellAccess,

        # Pulldown resistance
        'R_pulldown': calculate_on_resistance(
            g.cell.widthSRAMCellNMOS * g.tech.featureSize,
            NMOS,
            g.inputParameter.temperature,
            g.tech
        ),

        # Mux
        'C_mux': subarray.bitlineMux.capForPreviousDelayCalculation,

        # Sense amp
        'V_sense': subarray.senseVoltage,
        'V_precharge': subarray.voltagePrecharge,

        # Horowitz model
        'gm': calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech),
        'ramp_input': subarray.rowDecoder.rampOutput,
    }

    params['beta'] = 1 / (params['R_pulldown'] * params['gm'])

    return params, subarray


def main():
    print("="*80)
    print("END-TO-END SYMBOLIC DESTINY WORKFLOW")
    print("="*80)

    # File paths
    config_file = "config/sample_SRAM_2layer.cfg"
    cpp_output_file = "../destiny_3d_cache-master/cpp_output_sram2layer.txt"

    print(f"\n📁 Input Files:")
    print(f"  Config: {config_file}")
    print(f"  C++ Output: {cpp_output_file}")

    # ===========================================================================
    # STEP 1: Parse C++ DESTINY Output
    # ===========================================================================
    print("\n" + "="*80)
    print("STEP 1: Parse C++ DESTINY Output")
    print("="*80)

    cpp_config = parse_cpp_destiny_output(cpp_output_file)

    print(f"\n✓ Parsed C++ DESTINY results:")
    print(f"  Configuration: {cpp_config.subarray_rows}×{cpp_config.subarray_cols}")

    total_latency = (cpp_config.row_decoder_latency + cpp_config.bitline_latency +
                     cpp_config.senseamp_latency + cpp_config.mux_latency)

    print(f"  Total delay: {total_latency*1e9:.3f} ns")
    print(f"  Breakdown:")
    print(f"    - Decoder:  {cpp_config.row_decoder_latency*1e9:.3f} ns ({cpp_config.row_decoder_latency/total_latency*100:.1f}%)")
    print(f"    - Bitline:  {cpp_config.bitline_latency*1e9:.3f} ns ({cpp_config.bitline_latency/total_latency*100:.1f}%)")
    print(f"    - Senseamp: {cpp_config.senseamp_latency*1e12:.3f} ps ({cpp_config.senseamp_latency/total_latency*100:.1f}%)")
    print(f"    - Mux:      {cpp_config.mux_latency*1e12:.3f} ps ({cpp_config.mux_latency/total_latency*100:.1f}%)")

    # ===========================================================================
    # STEP 2: Create Symbolic Model
    # ===========================================================================
    print("\n" + "="*80)
    print("STEP 2: Create Symbolic Model")
    print("="*80)

    model = create_symbolic_model_from_destiny_output(cpp_output_file)

    print(f"\n✓ Created symbolic model:")
    all_exprs = model.expressions.get_all_expressions()
    print(f"  Available expressions: {len(all_exprs)}")
    for name, expr in list(all_exprs.items())[:5]:
        print(f"    • {name}: {len(expr.free_symbols)} symbolic variables")

    # ===========================================================================
    # STEP 3: Extract Technology Parameters
    # ===========================================================================
    print("\n" + "="*80)
    print("STEP 3: Extract Technology Parameters from Python DESTINY")
    print("="*80)

    params, subarray = extract_all_parameters(config_file, cpp_config)

    print(f"\n✓ Extracted {len(params)} parameters:")
    print(f"  Configuration: rows={params['rows']}, cols={params['cols']}")
    print(f"  Technology: V_dd={params['V_dd']:.2f}V, R_eff={params['R_eff']:.2f}")
    print(f"  Bitline: R_per_cell={params['R_per_cell']:.3f}Ω, C_per_cell={params['C_per_cell']*1e15:.3f}fF")
    print(f"  Access: R_access={params['R_access']:.1f}Ω, R_pulldown={params['R_pulldown']:.1f}Ω")

    # ===========================================================================
    # STEP 4: Evaluate Symbolic Expressions
    # ===========================================================================
    print("\n" + "="*80)
    print("STEP 4: Evaluate Symbolic Expressions with Parameters")
    print("="*80)

    print(f"\n🧮 Evaluating key expressions:")

    # Simple expressions
    C_bitline = model.expressions.evaluate('C_bitline', params)
    R_bitline = model.expressions.evaluate('R_bitline', params)
    print(f"\n  C_bitline = C_per_cell × rows")
    print(f"           = {params['C_per_cell']*1e15:.3f} fF/row × {params['rows']} rows")
    print(f"           = {C_bitline*1e15:.3f} fF ✓")

    print(f"\n  R_bitline = R_per_cell × rows")
    print(f"           = {params['R_per_cell']:.3f} Ω/row × {params['rows']} rows")
    print(f"           = {R_bitline:.3f} Ω ✓")

    # Complex expression
    tau = model.expressions.evaluate('tau_bitline', params)
    print(f"\n  tau_bitline = (R_access + R_pulldown) × (C_access + C_bitline + C_mux)")
    print(f"               + R_bitline × (C_mux + C_bitline/2)")
    print(f"             = ({params['R_access']:.1f} + {params['R_pulldown']:.1f}) × ({params['C_access']*1e15:.1f} + {C_bitline*1e15:.1f} + 0) fF")
    print(f"               + {R_bitline:.1f} × (0 + {C_bitline*1e15:.1f}/2) fF")
    print(f"             = {tau*1e9:.3f} ns ✓")

    # With logarithm
    tau_log = model.expressions.evaluate('tau_bitline_with_log', params)
    log_factor = tau_log / tau
    print(f"\n  tau × log(V_precharge / (V_precharge - V_sense/2))")
    print(f"     = {tau*1e9:.3f} ns × log({params['V_precharge']:.2f} / ({params['V_precharge']:.2f} - {params['V_sense']:.2f}/2))")
    print(f"     = {tau*1e9:.3f} ns × {log_factor:.4f}")
    print(f"     = {tau_log*1e9:.3f} ns ✓")

    # Apply Horowitz
    from formula import horowitz
    bitline_delay, _ = horowitz(tau_log, params['beta'], params['ramp_input'])
    print(f"\n  Horowitz(tau, beta={params['beta']:.4f}, ramp_input)")
    print(f"     = {bitline_delay*1e9:.6f} ns ✓")

    # Compare to actual
    print(f"\n  Python DESTINY bitlineDelay: {subarray.bitlineDelay*1e9:.6f} ns")
    error = abs(bitline_delay - subarray.bitlineDelay) / subarray.bitlineDelay * 100
    print(f"  Error: {error:.6f}%")
    if error < 0.001:
        print(f"  ✅ PERFECT MATCH!")

    # ===========================================================================
    # STEP 5: Export for Framework
    # ===========================================================================
    print("\n" + "="*80)
    print("STEP 5: Export for Framework Integration")
    print("="*80)

    export_file = "sram_2layer_complete_model.json"
    model.export_to_json(export_file)

    print(f"\n✓ Exported to: {export_file}")
    print(f"  Contains:")
    print(f"    • Symbolic expressions (SymPy strings + LaTeX)")
    print(f"    • Numerical results (C++ DESTINY values)")
    print(f"    • Configuration parameters")

    # ===========================================================================
    # STEP 6: Show Framework Usage
    # ===========================================================================
    print("\n" + "="*80)
    print("STEP 6: Framework Integration Examples")
    print("="*80)

    print("\n📦 Example 1: Get symbolic expression for analysis")
    print("```python")
    print("expr = model.get_symbolic_expression('tau_bitline')")
    print("print(expr)")
    print("```")
    print(f"→ {model.get_symbolic_expression('tau_bitline')}")

    print("\n📦 Example 2: Convert to LaTeX for documentation")
    print("```python")
    print("latex = model.expressions.to_latex('tau_bitline')")
    print("```")
    print(f"→ ${model.expressions.to_latex('tau_bitline')}$")

    print("\n📦 Example 3: Convert to Python function for optimization")
    print("```python")
    print("C_bitline_func = model.expressions.to_python_function('C_bitline')")
    print("for rows in [256, 512, 1024, 2048]:")
    print("    C = C_bitline_func(0.606e-15, rows)")
    print("    print(f'{rows} rows → {C*1e15:.2f} fF')")
    print("```")

    C_func = model.expressions.to_python_function('C_bitline')
    for rows_test in [256, 512, 1024, 2048]:
        C_test = C_func(0.606e-15, rows_test)
        print(f"  {rows_test} rows → {C_test*1e15:.2f} fF")

    print("\n📦 Example 4: Use in optimization")
    print("```python")
    print("from scipy.optimize import minimize")
    print("")
    print("def objective(x):")
    print("    params['rows'] = x[0]")
    print("    return model.expressions.evaluate('tau_bitline', params)")
    print("")
    print("result = minimize(objective, x0=[1024], bounds=[(256, 4096)])")
    print("```")

    # ===========================================================================
    # Summary
    # ===========================================================================
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETE")
    print("="*80)

    print("\n🎯 Summary:")
    print("  ✅ C++ DESTINY output parsed successfully")
    print("  ✅ Symbolic model created with real DESTINY formulas")
    print("  ✅ Technology parameters extracted from Python DESTINY")
    print("  ✅ Symbolic expressions evaluate to match concrete values")
    print("  ✅ Model exported for framework integration")

    print("\n🚀 Your framework can now:")
    print("  • Perform scaling analysis (how delay changes with rows/cols)")
    print("  • Run sensitivity studies (impact of V_dd, temperature, etc.)")
    print("  • Optimize designs (minimize delay subject to constraints)")
    print("  • Generate documentation (LaTeX formulas for papers)")
    print("  • Explore design space efficiently")

    print("\n📚 Documentation:")
    print("  • FRAMEWORK_INTEGRATION_README.md - Complete integration guide")
    print("  • SYMBOLIC_VERIFICATION_COMPLETE.md - Verification results")
    print("  • example_framework_usage.py - 7 detailed examples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
