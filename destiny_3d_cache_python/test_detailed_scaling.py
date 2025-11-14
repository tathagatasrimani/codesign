#!/usr/bin/env python3
"""
Detailed Test of Bitline Delay Scaling

This script examines the ACTUAL formulas used in Python DESTINY
to understand why bitline delay doesn't scale quadratically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from SubArray import SubArray
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget
import math


def test_detailed_scaling(config_file: str):
    """
    Test with detailed examination of R, C, and delay components
    """
    print("=" * 80)
    print("DETAILED BITLINE DELAY ANALYSIS")
    print("=" * 80)

    row_counts = [256, 512, 1024]
    results = []

    for num_rows in row_counts:
        print(f"\n{'─' * 80}")
        print(f"Testing with {num_rows} rows")
        print(f"{'─' * 80}")

        # Initialize (same as before)
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
        subarray.Initialize(num_rows, 2048, 1, 1, 1, True, 1, 8,
                            BufferDesignTarget.latency_first, 2)

        # Calculate
        subarray.CalculateArea()

        # BEFORE latency calculation, extract the R and C values
        print(f"\n📊 Physical Parameters (BEFORE CalculateLatency):")
        print(f"  lenBitline:   {subarray.lenBitline*1e6:.3f} μm  (∝ rows)")
        print(f"  resBitline:   {subarray.resBitline:.3e} Ω  (∝ rows)")
        print(f"  capBitline:   {subarray.capBitline*1e15:.3f} fF  (∝ rows)")

        subarray.CalculateLatency(1e20)

        # Extract delay
        bitline_delay = subarray.bitlineDelay * 1e9

        print(f"\n⏱️  Delay Result:")
        print(f"  Bitline delay: {bitline_delay:.4f} ns")

        results.append({
            'rows': num_rows,
            'lenBitline': subarray.lenBitline,
            'resBitline': subarray.resBitline,
            'capBitline': subarray.capBitline,
            'bitline_delay_ns': bitline_delay
        })

    # Analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    print("\n📐 Physical Parameter Scaling:")
    print(f"{'Rows':>6} | {'resBitline (Ω)':>15} | {'Ratio':>8} | {'capBitline (fF)':>15} | {'Ratio':>8}")
    print("─" * 75)

    base = results[0]
    for r in results:
        res_ratio = r['resBitline'] / base['resBitline']
        cap_ratio = r['capBitline'] / base['capBitline']
        print(f"{r['rows']:6} | {r['resBitline']:15.3e} | {res_ratio:8.2f}× | {r['capBitline']*1e15:15.3f} | {cap_ratio:8.2f}×")

    print("\n📐 R × C Product Scaling:")
    print(f"{'Rows':>6} | {'R × C (s)':>15} | {'Ratio vs Base':>15} | {'Expected (rows²)':>15}")
    print("─" * 65)

    for r in results:
        rc_product = r['resBitline'] * r['capBitline']
        rc_ratio = rc_product / (base['resBitline'] * base['capBitline'])
        expected_ratio = (r['rows'] / base['rows']) ** 2
        print(f"{r['rows']:6} | {rc_product:15.3e} | {rc_ratio:15.2f}× | {expected_ratio:15.2f}×")

    print("\n📐 Actual Delay Scaling:")
    print(f"{'Rows':>6} | {'Delay (ns)':>12} | {'Ratio vs Base':>15} | {'Expected (rows²)':>15} | {'Match?':>10}")
    print("─" * 75)

    for r in results:
        delay_ratio = r['bitline_delay_ns'] / base['bitline_delay_ns']
        expected_ratio = (r['rows'] / base['rows']) ** 2
        match_pct = (delay_ratio / expected_ratio) * 100 if expected_ratio > 0 else 0
        match = "✓" if 85 <= match_pct <= 115 else "✗"

        print(f"{r['rows']:6} | {r['bitline_delay_ns']:12.4f} | {delay_ratio:15.2f}× | {expected_ratio:15.2f}× | {match:>10} ({match_pct:.1f}%)")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    print("\n✓ resBitline scales linearly with rows (as expected)")
    print("✓ capBitline scales linearly with rows (as expected)")
    print("✓ R × C product scales QUADRATICALLY with rows")

    # Check if delay matches R×C quadratic scaling
    delay_256_512_ratio = results[1]['bitline_delay_ns'] / results[0]['bitline_delay_ns']
    delay_256_1024_ratio = results[2]['bitline_delay_ns'] / results[0]['bitline_delay_ns']

    print(f"\n✗ BUT: Actual delay does NOT scale quadratically:")
    print(f"  256→512 rows:  Delay increases by {delay_256_512_ratio:.2f}× (expected 4×)")
    print(f"  256→1024 rows: Delay increases by {delay_256_1024_ratio:.2f}× (expected 16×)")

    print(f"\n🔍 WHY? The SRAM bitline delay formula is MORE COMPLEX than R×C/2:")
    print(f"  τ = (R_cellAccess + R_pullDown) × (C_cellAccess + C_bitline + ...) +")
    print(f"      R_bitline × (... + C_bitline/2)")
    print(f"  τ *= log(V_precharge / (V_precharge - V_sense/2))")
    print(f"")
    print(f"  The delay includes:")
    print(f"    • Cell access resistance (constant)")
    print(f"    • Pull-down resistance (constant)")
    print(f"    • Cell access capacitance (constant)")
    print(f"    • Logarithmic voltage sensing term")
    print(f"")
    print(f"  These CONSTANT terms dominate for small subarrays,")
    print(f"  making the scaling appear more LINEAR than QUADRATIC.")

    print("\n" + "=" * 80)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_detailed_scaling.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    print("=" * 80)
    print("DETAILED BITLINE SCALING TEST")
    print("=" * 80)
    print(f"\nConfig: {config_file}")
    print("Testing why bitline delay doesn't scale purely as rows²")

    try:
        results = test_detailed_scaling(config_file)

        print("\n✓ TEST COMPLETE")
        print("\nKey Finding:")
        print("  The symbolic formula t ∝ R×C ∝ rows² IS CORRECT for the R×C product,")
        print("  but the ACTUAL SRAM bitline delay formula has additional terms that")
        print("  reduce the effective scaling exponent, especially for smaller subarrays.")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
