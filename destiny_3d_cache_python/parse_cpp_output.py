#!/usr/bin/env python3
"""
Parser for C++ DESTINY output to extract optimal configuration
"""

import re
from typing import Dict, Any


class OptimalConfiguration:
    """Stores the optimal configuration from C++ DESTINY DSE"""

    def __init__(self):
        # Bank organization
        self.num_banks_x = None
        self.num_banks_y = None
        self.num_stacks = None

        # Mat organization
        self.num_mats_x = None
        self.num_mats_y = None

        # Subarray size
        self.subarray_rows = None
        self.subarray_cols = None

        # Mux levels
        self.senseamp_mux = None
        self.output_mux_l1 = None
        self.output_mux_l2 = None

        # Timing results from C++
        self.read_latency = None
        self.write_latency = None
        self.tsv_latency = None
        self.htree_latency = None
        self.mat_latency = None
        self.predecoder_latency = None
        self.subarray_latency = None
        self.row_decoder_latency = None
        self.bitline_latency = None
        self.senseamp_latency = None
        self.mux_latency = None
        self.precharge_latency = None

        # Cell parameters
        self.cell_type = None
        self.cell_area = None
        self.cell_aspect_ratio = None
        self.access_transistor_width = None

        # Wire types
        self.local_wire_type = None
        self.global_wire_type = None


def parse_cpp_destiny_output(output_file: str) -> OptimalConfiguration:
    """
    Parse C++ DESTINY output file and extract optimal configuration

    Args:
        output_file: Path to C++ DESTINY output file

    Returns:
        OptimalConfiguration object with all parameters
    """
    config = OptimalConfiguration()

    with open(output_file, 'r') as f:
        content = f.read()

    # Parse CACHE DATA ARRAY section (or main array if not cache)
    data_section = re.search(r'CACHE DATA ARRAY DETAILS.*?(?=CACHE TAG ARRAY|$)',
                             content, re.DOTALL)

    if not data_section:
        # Try non-cache format
        data_section = re.search(r'CONFIGURATION.*?RESULT', content, re.DOTALL)

    if data_section:
        section_text = data_section.group(0)

        # Parse Bank Organization: X x Y x Z
        bank_match = re.search(r'Bank Organization:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)',
                               section_text)
        if bank_match:
            config.num_banks_x = int(bank_match.group(1))
            config.num_banks_y = int(bank_match.group(2))
            config.num_stacks = int(bank_match.group(3))

        # Parse Mat Organization: X x Y
        mat_match = re.search(r'Mat Organization:\s*(\d+)\s*x\s*(\d+)',
                             section_text)
        if mat_match:
            config.num_mats_x = int(mat_match.group(1))
            config.num_mats_y = int(mat_match.group(2))

        # Parse Subarray Size: ROWS Rows x COLS Columns
        subarray_match = re.search(r'Subarray Size\s*:\s*(\d+)\s*Rows\s*x\s*(\d+)\s*Columns',
                                  section_text)
        if subarray_match:
            config.subarray_rows = int(subarray_match.group(1))
            config.subarray_cols = int(subarray_match.group(2))

        # Parse Mux levels
        senseamp_mux = re.search(r'Senseamp Mux\s*:\s*(\d+)', section_text)
        if senseamp_mux:
            config.senseamp_mux = int(senseamp_mux.group(1))

        output_l1 = re.search(r'Output Level-1 Mux:\s*(\d+)', section_text)
        if output_l1:
            config.output_mux_l1 = int(output_l1.group(1))

        output_l2 = re.search(r'Output Level-2 Mux:\s*(\d+)', section_text)
        if output_l2:
            config.output_mux_l2 = int(output_l2.group(1))

        # Parse timing details
        read_lat = re.search(r'Read Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if read_lat:
            config.read_latency = parse_time_value(read_lat.group(1), read_lat.group(2))

        write_lat = re.search(r'Write Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if write_lat:
            config.write_latency = parse_time_value(write_lat.group(1), write_lat.group(2))

        # Parse detailed timing breakdown
        tsv_lat = re.search(r'TSV Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if tsv_lat:
            config.tsv_latency = parse_time_value(tsv_lat.group(1), tsv_lat.group(2))

        htree_lat = re.search(r'H-Tree Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if htree_lat:
            config.htree_latency = parse_time_value(htree_lat.group(1), htree_lat.group(2))

        mat_lat = re.search(r'Mat Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if mat_lat:
            config.mat_latency = parse_time_value(mat_lat.group(1), mat_lat.group(2))

        predec_lat = re.search(r'Predecoder Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if predec_lat:
            config.predecoder_latency = parse_time_value(predec_lat.group(1), predec_lat.group(2))

        subarray_lat = re.search(r'Subarray Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if subarray_lat:
            config.subarray_latency = parse_time_value(subarray_lat.group(1), subarray_lat.group(2))

        rowdec_lat = re.search(r'Row Decoder Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if rowdec_lat:
            config.row_decoder_latency = parse_time_value(rowdec_lat.group(1), rowdec_lat.group(2))

        bitline_lat = re.search(r'Bitline Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if bitline_lat:
            config.bitline_latency = parse_time_value(bitline_lat.group(1), bitline_lat.group(2))

        senseamp_lat = re.search(r'Senseamp Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if senseamp_lat:
            config.senseamp_latency = parse_time_value(senseamp_lat.group(1), senseamp_lat.group(2))

        mux_lat = re.search(r'Mux Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if mux_lat:
            config.mux_latency = parse_time_value(mux_lat.group(1), mux_lat.group(2))

        precharge_lat = re.search(r'Precharge Latency\s*=\s*([\d.]+)([pnmu]?s)', section_text)
        if precharge_lat:
            config.precharge_latency = parse_time_value(precharge_lat.group(1), precharge_lat.group(2))

        # Parse cell parameters
        cell_match = re.search(r'Memory Cell:\s*(\w+)', section_text)
        if cell_match:
            config.cell_type = cell_match.group(1)

        cell_area = re.search(r'Cell Area \(F\^2\)\s*:\s*([\d.]+)', section_text)
        if cell_area:
            config.cell_area = float(cell_area.group(1))

        aspect_ratio = re.search(r'Cell Aspect Ratio\s*:\s*([\d.]+)', section_text)
        if aspect_ratio:
            config.cell_aspect_ratio = float(aspect_ratio.group(1))

        # Parse wire types
        local_wire = re.search(r'Wire Type\s*:\s*(.+?)(?=\n)', section_text)
        if local_wire:
            config.local_wire_type = local_wire.group(1).strip()

    return config


def parse_time_value(value_str: str, unit: str) -> float:
    """Convert time string with unit to seconds"""
    value = float(value_str)

    unit_multipliers = {
        's': 1.0,
        'ms': 1e-3,
        'us': 1e-6,
        'μs': 1e-6,
        'ns': 1e-9,
        'ps': 1e-12
    }

    return value * unit_multipliers.get(unit, 1.0)


def print_configuration(config: OptimalConfiguration):
    """Print the parsed configuration"""
    print("=" * 80)
    print("OPTIMAL CONFIGURATION FROM C++ DESTINY DSE")
    print("=" * 80)

    print("\nBank Organization:")
    print(f"  Banks (X x Y x Stacks): {config.num_banks_x} x {config.num_banks_y} x {config.num_stacks}")

    print("\nMat Organization:")
    print(f"  Mats (X x Y): {config.num_mats_x} x {config.num_mats_y}")

    print("\nSubarray:")
    print(f"  Size: {config.subarray_rows} Rows x {config.subarray_cols} Columns")

    print("\nMux Levels:")
    print(f"  Senseamp Mux: {config.senseamp_mux}")
    print(f"  Output L1 Mux: {config.output_mux_l1}")
    print(f"  Output L2 Mux: {config.output_mux_l2}")

    print("\nTiming Breakdown (from C++ DESTINY):")
    if config.read_latency:
        print(f"  Total Read Latency: {config.read_latency*1e9:.3f} ns")
    if config.tsv_latency:
        print(f"    TSV Latency: {config.tsv_latency*1e12:.3f} ps")
    if config.htree_latency:
        print(f"    H-Tree Latency: {config.htree_latency*1e12:.3f} ps")
    if config.mat_latency:
        print(f"    Mat Latency: {config.mat_latency*1e9:.3f} ns")
    if config.predecoder_latency:
        print(f"      Predecoder: {config.predecoder_latency*1e12:.3f} ps")
    if config.subarray_latency:
        print(f"      Subarray: {config.subarray_latency*1e9:.3f} ns")
    if config.row_decoder_latency:
        print(f"        Row Decoder: {config.row_decoder_latency*1e9:.3f} ns")
    if config.bitline_latency:
        print(f"        Bitline: {config.bitline_latency*1e9:.3f} ns")
    if config.senseamp_latency:
        print(f"        Senseamp: {config.senseamp_latency*1e12:.3f} ps")
    if config.mux_latency:
        print(f"        Mux: {config.mux_latency*1e12:.3f} ps")

    print("\nCell Parameters:")
    print(f"  Type: {config.cell_type}")
    print(f"  Area: {config.cell_area} F^2")
    print(f"  Aspect Ratio: {config.cell_aspect_ratio}")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_cpp_output.py <cpp_destiny_output_file>")
        sys.exit(1)

    config = parse_cpp_destiny_output(sys.argv[1])
    print_configuration(config)
