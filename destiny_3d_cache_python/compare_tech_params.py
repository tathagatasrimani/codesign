#!/usr/bin/env python3
"""
Compare Technology Parameters: C++ vs Python DESTINY
"""

print("="*80)
print("TECHNOLOGY PARAMETER COMPARISON")
print("65nm LOP @ 350K")
print("="*80)

# C++ DESTINY values (from Technology.cpp lines 729-763)
cpp_params = {
    'vdd': 0.8,
    'vth': 323.75e-3,
    'phyGateLength': 0.032e-6,
    'capIdealGate': 6.01e-10,
    'capFringe': 2.4e-10,
    'capJunction': 1.00e-3,
    'capOx': 1.88e-2,
    'effectiveElectronMobility': 491.59e-4,
    'effectiveHoleMobility': 110.95e-4,
    'pnSizeRatio': 2.28,
    'effectiveResistanceMultiplier': 1.82,
    'currentOnNmos_350K': 524.5,  # index [50] for 350K
    'currentOnPmos_350K': 299.8,
}

# Python DESTINY values (from Technology.py lines 436-455)
python_params = {
    'vdd': 0.8,
    'vth': 323.75e-3,
    'phyGateLength': 0.032e-6,
    'capIdealGate': 6.01e-10,
    'capFringe': 2.4e-10,
    'capJunction': 1.00e-3,
    'capOx': 1.88e-2,
    'effectiveElectronMobility': 491.59e-4,
    'effectiveHoleMobility': 110.95e-4,
    'pnSizeRatio': 2.28,
    'effectiveResistanceMultiplier': 1.82,
    'currentOnNmos_350K': 562.9,  # index [50] for 350K
    'currentOnPmos_350K': 329.5,
}

print("\nParameter Comparison:")
print(f"{'Parameter':<35} {'C++':<20} {'Python':<20} {'Match'}")
print("-"*80)

all_match = True
for key in cpp_params.keys():
    cpp_val = cpp_params[key]
    py_val = python_params[key]
    match = abs(cpp_val - py_val) < 1e-15
    match_str = "✓" if match else "✗ DIFFER"

    if not match:
        all_match = False
        ratio = py_val / cpp_val if cpp_val != 0 else 0
        print(f"{key:<35} {cpp_val:<20.6e} {py_val:<20.6e} {match_str} ({ratio:.3f}×)")
    else:
        print(f"{key:<35} {cpp_val:<20.6e} {py_val:<20.6e} {match_str}")

print("\n" + "="*80)
if all_match:
    print("✓ ALL PARAMETERS MATCH")
else:
    print("❌ PARAMETERS DIFFER")
    print("\nKey difference: currentOnNmos differs by factor:")
    print(f"  Python/C++ = {python_params['currentOnNmos_350K']/cpp_params['currentOnNmos_350K']:.3f}×")
    print(f"\nThis explains part of the 2× discrepancy!")
