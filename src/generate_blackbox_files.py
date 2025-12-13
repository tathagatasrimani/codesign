#!/usr/bin/env python3
"""
Script to generate JSON and Verilog files for blackbox functions from a C++ file.
"""

import re
import json
import sys
import os
from pathlib import Path
import logging
import math
from src import sim_util

logger = logging.getLogger(__name__)


def get_type_width(cpp_type):
    """Map C++ type to Verilog bit width."""
    type_map = {
        'float': 32,
        'double': 64,
        'int': 32,
        'unsigned int': 32,
        'long': 64,
        'unsigned long': 64,
        'short': 16,
        'unsigned short': 16,
        'char': 8,
        'unsigned char': 8,
    }
    return type_map.get(cpp_type.strip(), 32)


def parse_blackbox_functions(cpp_file):
    """Parse blackbox functions from a C++ file."""
    with open(cpp_file, 'r') as f:
        content = f.read()
    
    functions = []
    
    # Find all blackbox function markers
    # Pattern: // This is a blackbox function.\n followed by function definition
    pattern = r'//\s*This is a blackbox function\.\s*\n((?:[^\n]+\n)*?)(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+return\s+\w+;[^}]*)\}'
    
    matches = re.finditer(pattern, content, re.MULTILINE)
    
    for match in matches:
        func_info = {}
        sig_lines = match.group(1)
        return_type = match.group(2)
        func_name = match.group(3)
        params_str = match.group(4).strip()
        body = match.group(5)
        
        func_info['name'] = func_name
        func_info['return_type'] = return_type
        
        # Parse parameters
        params = []
        if params_str:
            # Split parameters by comma, handling multi-line
            param_parts = re.split(r',\s*(?=\w+\s+\w)', params_str.replace('\n', ' '))
            for param_str in param_parts:
                param_str = param_str.strip()
                if param_str:
                    # Match: type name
                    param_match = re.match(r'(\w+(?:\s+\w+)?)\s+(\w+)', param_str)
                    if param_match:
                        param_type = param_match.group(1).strip()
                        param_name = param_match.group(2).strip()
                        params.append({
                            'type': param_type,
                            'name': param_name
                        })
        
        func_info['params'] = params
        
        # Find return variable and assignment
        return_match = re.search(r'return\s+(\w+);', body)
        if return_match:
            return_var = return_match.group(1)
            func_info['return_var'] = return_var
            
            # Find assignment: type return_var = expression;
            assign_pattern = rf'{return_var}\s*=\s*([^;]+);'
            assign_match = re.search(assign_pattern, body)
            if assign_match:
                assignment = assign_match.group(1).strip()
                func_info['assignment'] = assignment
                
                # Determine operation type for description
                if '+' in assignment and '*' not in assignment and '/' not in assignment:
                    func_info['op_desc'] = 'addition'
                elif '*' in assignment:
                    func_info['op_desc'] = 'multiplication'
                elif '-' in assignment and '*' not in assignment and '/' not in assignment:
                    func_info['op_desc'] = 'subtraction'
                elif '/' in assignment:
                    func_info['op_desc'] = 'division'
                else:
                    func_info['op_desc'] = 'operation'
            else:
                func_info['assignment'] = return_var
                func_info['op_desc'] = 'operation'
            
            functions.append(func_info)
    
    # If regex didn't work, try line-by-line parsing
    if not functions:
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if '// This is a blackbox function.' in line:
                func_info = {}
                
                # Collect function signature
                sig_lines = []
                j = i + 1
                while j < len(lines) and '{' not in lines[j]:
                    if lines[j].strip():
                        sig_lines.append(lines[j].strip())
                    j += 1
                    if j >= len(lines):
                        break
                
                if j >= len(lines):
                    i += 1
                    continue
                
                # Parse signature
                sig_text = ' '.join(sig_lines)
                match = re.search(r'(\w+)\s+(\w+)\s*\(([^)]*)\)', sig_text)
                if match:
                    return_type = match.group(1)
                    func_name = match.group(2)
                    params_str = match.group(3).strip()
                    
                    func_info['name'] = func_name
                    func_info['return_type'] = return_type
                    
                    # Parse parameters
                    params = []
                    if params_str:
                        param_parts = re.split(r',\s*(?=\w+\s+\w)', params_str)
                        for param_str in param_parts:
                            param_str = param_str.strip()
                            if param_str:
                                param_match = re.match(r'(\w+(?:\s+\w+)?)\s+(\w+)', param_str)
                                if param_match:
                                    params.append({
                                        'type': param_match.group(1).strip(),
                                        'name': param_match.group(2).strip()
                                    })
                    
                    func_info['params'] = params
                    
                    # Parse body
                    body_lines = []
                    brace_count = 1
                    k = j
                    while k < len(lines) and brace_count > 0:
                        body_lines.append(lines[k].rstrip())
                        brace_count += lines[k].count('{')
                        brace_count -= lines[k].count('}')
                        k += 1
                    
                    body_text = '\n'.join(body_lines)
                    
                    # Find return
                    return_match = re.search(r'return\s+(\w+);', body_text)
                    if return_match:
                        return_var = return_match.group(1)
                        func_info['return_var'] = return_var
                        
                        functions.append(func_info)
                        i = k
                        continue
            
            i += 1
    
    return functions


def generate_json(func_info, cpp_filename, output_dir, latency, clk_period):
    """Generate JSON file for a blackbox function."""
    module_name = sim_util.get_module_map()[func_info['name']]
    num_cycles = math.ceil(latency[module_name] / clk_period)
    json_data = {
        "c_function_name": func_info['name'],
        "rtl_top_module_name": func_info['name'],
        "c_files": [
            {
                "c_file": cpp_filename,
                "cflag": ""
            }
        ],
        "rtl_files": [
            f"{os.path.join(output_dir, func_info['name'])}.v"
        ],
        "c_parameters": [],
        "c_return": {
            "c_port_direction": "out",
            "rtl_ports": {
                "data_write_out": func_info['return_var']
            }
        },
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "",
            "ap_ctrl_chain_protocol_start": "",
            "ap_ctrl_chain_protocol_ready": "",
            "ap_ctrl_chain_protocol_done": "",
            "ap_ctrl_chain_protocol_continue": ""
        },
        "rtl_performance": {
            "latency": f"{num_cycles}",
            "II": "1"
        },
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "0",
            "URAM": "0",
            "DSP": "1"
        }
    }
    
    # Add parameters
    for param in func_info['params']:
        json_data["c_parameters"].append({
            "c_name": param['name'],
            "c_port_direction": "in",
            "rtl_ports": {
                "data_read_in": param['name']
            }
        })
    
    # Write JSON file
    json_path = os.path.join(output_dir, f"{func_info['name']}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Generated: {json_path}")
    return json_path


def generate_verilog(func_info, output_dir):
    """Generate Verilog file for a blackbox function."""
    type_width = get_type_width(func_info['return_type'])
    
    verilog_lines = [
        f"module {func_info['name']} ("
    ]
    
    # Add input parameters
    for param in func_info['params']:
        param_width = get_type_width(param['type'])
        verilog_lines.append(f"    input wire [{param_width-1}:0] {param['name']},")
    
    # Add output
    verilog_lines.append(f"    output wire [{type_width-1}:0] {func_info['return_var']},")
    
    # Add common signals
    verilog_lines.extend([
        "    // Common signals",
        "    input wire ap_clk,",
        "    input wire ap_rst,",
        "    input wire ap_ce,",
        ");",
        "",
        f"    assign {func_info['return_var']} = {func_info['assignment']};",
        "",
        "endmodule",
        ""
    ])
    
    # Write Verilog file
    verilog_path = os.path.join(output_dir, f"{func_info['name']}.v")
    with open(verilog_path, 'w') as f:
        f.write('\n'.join(verilog_lines))
    
    logger.info(f"Generated: {verilog_path}")
    return verilog_path

def generate_blackbox_files(cpp_file, output_dir, tcl_file, latency, clk_period):
    if not os.path.exists(cpp_file):
        logger.error(f"Error: C++ file not found: {cpp_file}")
        sys.exit(1)
    
    # Parse blackbox functions
    logger.info(f"Parsing blackbox functions from: {cpp_file}")
    functions = parse_blackbox_functions(cpp_file)

    logger.info(f"Found {len(functions)} blackbox function(s):")
    for func in functions:
        logger.info(f"  - {func['name']}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate files for each function
    for func in functions:
        generate_json(func, cpp_file, output_dir, latency, clk_period)
        generate_verilog(func, output_dir)
    
    logger.info(f"\nSuccessfully generated files in: {output_dir}")
    f = open(tcl_file, 'r')
    tcl_content = f.readlines()
    newlines = []
    already_added = False
    for line in tcl_content:
        newlines.append(line)
        if "add_files" in line and not already_added:
            for func in functions:
                filepath = os.path.join(output_dir, f"{func['name']}.json")
                newlines.append(f"add_files -blackbox {filepath}\n")
            already_added = True
    with open(tcl_file, 'w') as f:
        f.writelines(newlines)


def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python3 generate_blackbox_files.py <cpp_file> [output_dir]")
        sys.exit(1)
    
    cpp_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(cpp_file)
    
    if not os.path.exists(cpp_file):
        logger.info(f"Error: C++ file not found: {cpp_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse blackbox functions
    logger.info(f"Parsing blackbox functions from: {cpp_file}")
    functions = parse_blackbox_functions(cpp_file)
    
    if not functions:
        logger.info("No blackbox functions found!")
        sys.exit(1)
    
    logger.info(f"Found {len(functions)} blackbox function(s):")
    for func in functions:
        logger.info(f"  - {func['name']}")
    
    # Generate files for each function
    for func in functions:
        generate_json(func, cpp_file, output_dir)
        generate_verilog(func, output_dir)
    
    logger.info(f"\nSuccessfully generated files in: {output_dir}")


if __name__ == '__main__':
    main()
