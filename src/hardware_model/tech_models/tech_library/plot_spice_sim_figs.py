#!/usr/bin/env python3
"""Generate plots from existing SPICE simulation data without re-running simulations."""

import os
from pathlib import Path

# Import data extraction and plotting functions from analyze modules
import sys
benchmarks_path = Path(__file__).parent.parent / 'mvs-transistor-models' / 'benchmarks'
sys.path.insert(0, str(benchmarks_path))

from analyze_fo4 import extract_dc_data as extract_fo4_dc_data, plot_vtc
from analyze_transistor import extract_dc_data as extract_transistor_dc_data, find_signal, plot_iv_curves_simple
from analyze_ring import extract_tran_data, plot_ring_waveform, measure_ring_frequency
import numpy as np


def plot_fo4_from_raw(sim_folder, output_file=None):
    """Generate FO4 VTC plot from existing raw data.

    Args:
        sim_folder: Path to simulation folder containing fo4_inverter.raw/
        output_file: Output path for plot (default: sim_folder/fo4_inverter.png)

    Returns:
        dict with vin, vout1, vout2 arrays
    """
    sim_folder = Path(sim_folder)
    raw_dir = sim_folder / 'fo4_inverter.raw'
    assert raw_dir.exists(), f"FO4 raw data not found: {raw_dir}"

    dc_data = extract_fo4_dc_data(raw_dir)
    assert dc_data is not None, "Failed to extract FO4 DC data"

    vin = dc_data['vin']
    vout1 = dc_data['vout1']
    vout2 = dc_data.get('vout2')

    output_file = output_file or (sim_folder / 'fo4_inverter.png')
    plot_vtc(vin, vout1, vout2, output_file=output_file)

    return {'vin': vin, 'vout1': vout1, 'vout2': vout2}


def plot_transistor_iv_from_raw(sim_folder, output_file=None):
    """Generate transistor IV curves plot from existing raw data.

    Args:
        sim_folder: Path to simulation folder containing single_transistor.raw/
        output_file: Output path for plot (default: sim_folder/single_transistor.png)

    Returns:
        dict with vgs, ids_vgs, vds_values
    """
    sim_folder = Path(sim_folder)
    raw_dir = sim_folder / 'single_transistor.raw'
    assert raw_dir.exists(), f"Transistor raw data not found: {raw_dir}"

    dc_data = extract_transistor_dc_data(raw_dir)
    assert dc_data is not None, "Failed to extract transistor DC data"

    # Find all Vgs sweep files (dc_vgs_vds0, dc_vgs_vds1, etc.)
    vgs_sweeps = sorted([k for k in dc_data.keys() if 'dc_vgs' in k.lower()])
    assert len(vgs_sweeps) > 0, f"No Vgs sweeps found in {list(dc_data.keys())}"

    # Extract data from each sweep
    vgs = None
    ids_by_idx = {}

    for sweep_name in vgs_sweeps:
        data = dc_data[sweep_name]
        v_sig = find_signal(data, ['vgs', 'g'])
        i_sig = find_signal(data, [':d', ':id', 'id', 'i('])

        assert v_sig is not None, f"No voltage signal in {sweep_name}"
        assert i_sig is not None, f"No current signal in {sweep_name}"

        if vgs is None:
            vgs = data[v_sig]

        idx = int(sweep_name.split('vds')[-1]) if 'vds' in sweep_name else 0
        ids_by_idx[idx] = np.abs(data[i_sig])

    assert vgs is not None, "No Vgs data extracted"

    sorted_indices = sorted(ids_by_idx.keys())
    n_curves = len(sorted_indices)
    vdd = np.max(vgs)
    vds_values = [vdd * (i + 1) / n_curves for i in sorted_indices]
    ids_vgs = np.array([ids_by_idx[i] for i in sorted_indices])

    output_file = output_file or (sim_folder / 'single_transistor.png')
    plot_iv_curves_simple(vgs, ids_vgs, vds_values, output_file)

    return {'vgs': vgs, 'ids_vgs': ids_vgs, 'vds_values': vds_values}


def plot_ring_from_raw(sim_folder, output_file=None, num_stages=7, num_cycles=5):
    """Generate ring oscillator waveform plot from existing raw data.

    Args:
        sim_folder: Path to simulation folder containing ring_oscillator.raw/
        output_file: Output path for plot (default: sim_folder/ring_waveform.png)
        num_stages: Number of inverter stages (default: 7)
        num_cycles: Number of cycles to plot (default: 5)

    Returns:
        dict with time, v, frequency data
    """
    sim_folder = Path(sim_folder)
    raw_dir = sim_folder / 'ring_oscillator.raw'
    assert raw_dir.exists(), f"Ring oscillator raw data not found: {raw_dir}"

    tran_data = extract_tran_data(raw_dir, node_name='n0')
    assert tran_data is not None, "Failed to extract ring oscillator transient data"

    time = tran_data['time']
    v = tran_data['n0']

    # Measure frequency
    freq_data = measure_ring_frequency(time, v, num_stages=num_stages)
    period = freq_data['T_period_ps'] * 1e-12

    output_file = output_file or (sim_folder / 'ring_waveform.png')
    plot_ring_waveform(time, v, output_file, period=period, num_cycles=num_cycles)

    return {'time': time, 'v': v, **freq_data}


def generate_all_plots(sim_folder, num_stages=7):
    """Generate all plots from a SPICE simulation folder.

    Args:
        sim_folder: Path to simulation folder containing .raw directories
        num_stages: Number of ring oscillator stages (default: 7)

    Returns:
        dict with results from each plot type
    """
    sim_folder = Path(sim_folder)
    results = {}

    # FO4 inverter
    fo4_raw = sim_folder / 'fo4_inverter.raw'
    if fo4_raw.exists():
        results['fo4'] = plot_fo4_from_raw(sim_folder)
        print(f"Generated: {sim_folder / 'fo4_inverter.png'}")

    # Transistor IV
    transistor_raw = sim_folder / 'single_transistor.raw'
    if transistor_raw.exists():
        results['transistor'] = plot_transistor_iv_from_raw(sim_folder)
        print(f"Generated: {sim_folder / 'single_transistor.png'}")

    # Ring oscillator
    ring_raw = sim_folder / 'ring_oscillator.raw'
    if ring_raw.exists():
        results['ring'] = plot_ring_from_raw(sim_folder, num_stages=num_stages)
        print(f"Generated: {sim_folder / 'ring_waveform.png'}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate plots from SPICE simulation data')
    parser.add_argument('sim_folder', type=Path, help='Path to simulation folder')
    parser.add_argument('--num-stages', type=int, default=7, help='Ring oscillator stages')

    args = parser.parse_args()
    generate_all_plots(args.sim_folder, num_stages=args.num_stages)
