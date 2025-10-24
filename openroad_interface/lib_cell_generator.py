#!/usr/bin/env python3
"""
LIB Cell Generator

This script generates LIB (Library) cell definitions for standard cells and macros.
It creates cells with customizable input pins, output pins, timing parameters, and other properties.

Usage:
    python lib_cell_generator.py

Example:
    # Generate a simple AND gate
    generator = LibCellGenerator()
    cell_def = generator.generate_cell(
        cell_name="AND2",
        input_pins=["A", "B"],
        output_pins=["Y"],
        delay=50.0,
        leakage=1000.0,
        area=100.0
    )
    print(cell_def)
"""

import argparse
import sys
from typing import List, Dict, Any, Optional


class LibCellGenerator:
    """Generator for LIB cell definitions."""
    
    def __init__(self):
        self.default_drive_strength = 1
        self.default_rise_transition = 0.1
        self.default_fall_transition = 0.1
        self.default_timing_sense = "positive_unate"
        self.power_pin = "VDD"
        self.ground_pin = "VSS"
    
    def generate_cell(self, 
                     cell_name: str,
                     input_pins: List[str],
                     output_pins: List[str],
                     delay: float,
                     leakage: float,
                     area: float,
                     drive_strength: Optional[int] = None,
                     rise_transition: Optional[float] = None,
                     fall_transition: Optional[float] = None,
                     timing_sense: Optional[str] = None,
                     power_pin: Optional[str] = None,
                     ground_pin: Optional[str] = None,
                     output_timing_relations: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Generate a complete LIB cell definition.
        
        Args:
            cell_name: Name of the cell (e.g., "AND2", "Mult64_40_2")
            input_pins: List of input pin names (e.g., ["A", "B"])
            output_pins: List of output pin names (e.g., ["Y"])
            delay: Delay value for cell_rise and cell_fall (e.g., 150.0)
            leakage: Cell leakage power (e.g., 108640.1)
            area: Cell area (e.g., 8340.2144)
            drive_strength: Drive strength (default: 1)
            rise_transition: Rise transition time (default: 0.1)
            fall_transition: Fall transition time (default: 0.1)
            timing_sense: Timing sense (default: "positive_unate")
            power_pin: Power pin name (default: "VDD")
            ground_pin: Ground pin name (default: "VSS")
            output_timing_relations: Dict mapping output pins to their related input pins
                                    (default: each output relates to all inputs)
        
        Returns:
            Complete LIB cell definition as a string
        """
        # Set defaults
        drive_strength = drive_strength or self.default_drive_strength
        rise_transition = rise_transition or self.default_rise_transition
        fall_transition = fall_transition or self.default_fall_transition
        timing_sense = timing_sense or self.default_timing_sense
        power_pin = power_pin or self.power_pin
        ground_pin = ground_pin or self.ground_pin
        
        # Default timing relations: each output relates to all inputs
        if output_timing_relations is None:
            output_timing_relations = {pin: input_pins for pin in output_pins}
        
        # Generate cell definition
        cell_def = []
        
        # Cell header
        cell_def.append(f"\tcell ({cell_name}) {{")
        cell_def.append(f"\t\tdrive_strength : {drive_strength};")
        cell_def.append(f"\t\tarea : {area};")
        cell_def.append("")
        
        # Power and ground pins
        cell_def.append(f"\t\tpg_pin({power_pin}) {{")
        cell_def.append(f"\t\t\tvoltage_name : {power_pin};")
        cell_def.append(f"\t\t\tpg_type      : primary_power;")
        cell_def.append(f"\t\t}}")
        cell_def.append(f"\t\tpg_pin({ground_pin}) {{")
        cell_def.append(f"\t\t\tvoltage_name : {ground_pin};")
        cell_def.append(f"\t\t\tpg_type      : primary_ground;")
        cell_def.append(f"\t\t}}")
        cell_def.append("")
        
        # Cell leakage power
        cell_def.append(f"\t\tcell_leakage_power : {leakage};")
        cell_def.append("")
        
        # Input pins
        for pin in input_pins:
            cell_def.append(f"\t\tpin ({pin}) {{")
            cell_def.append(f"\t\t\tdirection : input;")
            cell_def.append(f'\t\t\trelated_power_pin\t: "{power_pin}";')
            cell_def.append(f'\t\t\trelated_ground_pin\t: "{ground_pin}";')
            cell_def.append(f"\t\t}}")
        
        cell_def.append("")
        
        # Output pins with timing
        for output_pin in output_pins:
            cell_def.append(f"\t\tpin ({output_pin}) {{")
            cell_def.append(f"\t\t\tdirection: output;")
            
            # Generate timing blocks for each related input pin
            related_inputs = output_timing_relations.get(output_pin, input_pins)
            for input_pin in related_inputs:
                cell_def.append(f"\t\t\ttiming () {{")
                cell_def.append(f'\t\t\t\trelated_pin : "{input_pin}";')
                cell_def.append(f"\t\t\t\ttiming_sense : {timing_sense};")
                cell_def.append("")
                cell_def.append(f"\t\t\t\tcell_rise (scalar) {{ values ( {delay} ); }}")
                cell_def.append(f"\t\t\t\tcell_fall (scalar) {{ values ( {delay} ); }}")
                cell_def.append(f"\t\t\t\trise_transition (scalar) {{ values ( {rise_transition} ); }}")
                cell_def.append(f"\t\t\t\tfall_transition (scalar) {{ values ( {fall_transition} ); }}")
                cell_def.append(f"\t\t\t}}")
            
            cell_def.append(f"\t\t}}")
        
        # Cell footer
        cell_def.append("")
        cell_def.append("\t}")
        
        return "\n".join(cell_def)
    
    def generate_multiple_cells(self, macro_dict: Dict[str, Any], circuit_model) -> str:
        """
        Generate multiple cells and return them as a complete LIB file section.
        
        Args:
            cell_specs: List of dictionaries, each containing cell parameters
        
        Returns:
            Complete LIB file section with multiple cells
        """
        cell_specs = []
        for macro_name, macro_data in macro_dict.items():
            if "function" not in macro_data:
                continue
            cell_specs.append({
                "cell_name": macro_name,
                "input_pins": macro_data["input"],
                "output_pins": macro_data["output"],
                "delay": circuit_model.symbolic_latency_wc[macro_data["function"]]().xreplace(circuit_model.tech_model.base_params.tech_values).evalf(),
                "leakage": circuit_model.symbolic_power_passive[macro_data["function"]]().xreplace(circuit_model.tech_model.base_params.tech_values).evalf() * 1e-9, # convert from W to nW
                "area": macro_data["area"]
            })
        cells = []
        for spec in cell_specs:
            cell_def = self.generate_cell(**spec)
            cells.append(cell_def)
        
        return "\n".join(cells)
    
    def write_lib_file(self, cells: str, output_file: str):
        """
        Append generated cells to an existing LIB file before the last closing bracket.
        
        Args:
            cells: Generated cell definitions as a string
            output_file: Path to the LIB file to append to
        """
        # Read the existing LIB file
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Find the last closing bracket of the library definition
        # The structure is: library (codesign) { ... cells ... }
        # We need to insert before the final }
        
        # Split the content to find where to insert
        lines = content.split('\n')
        
        # Find the last closing bracket (should be around line 4109)
        last_closing_bracket_line = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == '}':
                last_closing_bracket_line = i
                break
        
        if last_closing_bracket_line == -1:
            raise ValueError("Could not find the last closing bracket in the LIB file")
        
        # Insert the new cells before the last closing bracket
        # Add some spacing for readability
        new_lines = lines[:last_closing_bracket_line]
        new_lines.append("")  # Empty line for spacing
        new_lines.extend(cells.split('\n'))
        new_lines.append("")  # Empty line for spacing
        new_lines.extend(lines[last_closing_bracket_line:])  # Keep the rest
        
        # Write the modified content back to the file
        with open(output_file, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print(f"Successfully appended {len(cells.split('cell (')) - 1} cells to {output_file}")
    
    def generate_and_write_cells(self, macro_dict: Dict[str, Any], circuit_model, output_file: str):
        """
        Generate cells from macro dictionary and circuit model, then append to LIB file.
        
        Args:
            macro_dict: Dictionary containing macro definitions
            circuit_model: Circuit model for timing and power calculations
            output_file: Path to the LIB file to append to
        """
        # Generate the cells
        cells = self.generate_multiple_cells(macro_dict, circuit_model)
        
        # Write to the LIB file
        self.write_lib_file(cells, output_file)


def main():
    """Command-line interface for the LIB cell generator."""
    parser = argparse.ArgumentParser(description="Generate LIB cell definitions")
    parser.add_argument("--cell-name", required=True, help="Name of the cell")
    parser.add_argument("--input-pins", nargs="+", required=True, help="Input pin names")
    parser.add_argument("--output-pins", nargs="+", required=True, help="Output pin names")
    parser.add_argument("--delay", type=float, required=True, help="Delay value for timing")
    parser.add_argument("--leakage", type=float, required=True, help="Cell leakage power")
    parser.add_argument("--area", type=float, required=True, help="Cell area")
    parser.add_argument("--drive-strength", type=int, default=1, help="Drive strength")
    parser.add_argument("--rise-transition", type=float, default=0.1, help="Rise transition time")
    parser.add_argument("--fall-transition", type=float, default=0.1, help="Fall transition time")
    parser.add_argument("--timing-sense", default="positive_unate", help="Timing sense")
    parser.add_argument("--power-pin", default="VDD", help="Power pin name")
    parser.add_argument("--ground-pin", default="VSS", help="Ground pin name")
    parser.add_argument("--output-file", help="Output file path (if not specified, prints to stdout)")
    
    args = parser.parse_args()
    
    # Create generator and generate cell
    generator = LibCellGenerator()
    cell_def = generator.generate_cell(
        cell_name=args.cell_name,
        input_pins=args.input_pins,
        output_pins=args.output_pins,
        delay=args.delay,
        leakage=args.leakage,
        area=args.area,
        drive_strength=args.drive_strength,
        rise_transition=args.rise_transition,
        fall_transition=args.fall_transition,
        timing_sense=args.timing_sense,
        power_pin=args.power_pin,
        ground_pin=args.ground_pin
    )
    
    # Output result
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(cell_def)
        print(f"Cell definition written to {args.output_file}")
    else:
        print(cell_def)


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:  # No command line arguments
        print("LIB Cell Generator - Example Usage")
        print("=" * 50)
        
        generator = LibCellGenerator()
        
        # Example 1: Simple AND gate
        print("Example 1: AND2 gate")
        print("-" * 30)
        and2_cell = generator.generate_cell(
            cell_name="AND2",
            input_pins=["A", "B"],
            output_pins=["Y"],
            delay=50.0,
            leakage=1000.0,
            area=100.0
        )
        print(and2_cell)
        print()
        
        # Example 2: Multi-bit multiplier (like Mult64_40_2)
        print("Example 2: Multi-bit multiplier")
        print("-" * 30)
        input_pins = [f"A{i}" for i in range(16)] + [f"B{i}" for i in range(16)]
        output_pins = [f"Z{i}" for i in range(16)]
        
        mult_cell = generator.generate_cell(
            cell_name="Mult32_16",
            input_pins=input_pins,
            output_pins=output_pins,
            delay=150.0,
            leakage=50000.0,
            area=5000.0
        )
        print(mult_cell)
        print()
        
        # Example 3: Custom timing relations
        print("Example 3: Custom timing relations")
        print("-" * 30)
        custom_cell = generator.generate_cell(
            cell_name="CustomGate",
            input_pins=["A", "B", "C"],
            output_pins=["Y1", "Y2"],
            delay=75.0,
            leakage=2000.0,
            area=200.0,
            output_timing_relations={
                "Y1": ["A", "B"],  # Y1 depends on A and B
                "Y2": ["B", "C"]   # Y2 depends on B and C
            }
        )
        print(custom_cell)
        
    else:
        main()
