import sys
import os
import re
import json


def extract_sections(filename, output_folder="."):
    netlist_lines = []
    complist_lines = []
    fsm_lines = []
    transitions = {}

    inside_netlist = False
    inside_complist = False
    inside_fsm = False
    inside_transitions = False
    inside_stg = False
    stg_lines = []

    stg_start = "---------------- STG Properties BEGIN ----------------"
    stg_end = "---------------- STG Properties END ------------------"

    with open(filename, 'r') as f:
        for line in f:
            if '<net_list>' in line:
                inside_netlist = True
                netlist_lines.append(line)
                continue
            if '</net_list>' in line:
                netlist_lines.append(line)
                inside_netlist = False
                continue
            if inside_netlist:
                netlist_lines.append(line)
                continue

            if '<comp_list>' in line:
                inside_complist = True
                complist_lines.append(line)
                continue
            if '</comp_list>' in line:
                complist_lines.append(line)
                inside_complist = False
                continue
            if inside_complist:
                complist_lines.append(line)
                continue

            # Collect FSM state operations lines
            if '* FSM state operations:' in line:
                inside_fsm = True
                fsm_lines.append(line)
                continue
            if '============================================================' in line:
                inside_fsm = False
                continue
            if inside_fsm:
                fsm_lines.append(line)
                continue

            # Collect FSM state transitions lines and parse them
            if '* FSM state transitions:' in line:
                inside_transitions = True
                continue
            if inside_transitions:
                l = line.strip()
                if not l:
                    continue
                if '-->' in l:
                    parts = l.split('-->')
                    if len(parts) != 2:
                        continue
                    from_state = parts[0].strip()
                    to_state = parts[1].strip()
                    if from_state.isdigit():
                        if to_state == '':
                            transitions[int(from_state)] = -1
                        elif to_state.isdigit():
                            transitions[int(from_state)] = int(to_state)
                else:
                    # End of transitions section if line doesn't match
                    inside_transitions = False

            # Collect STG Properties section
            if stg_start in line:
                inside_stg = True
                continue
            if stg_end in line:
                inside_stg = False
                continue
            if inside_stg:
                stg_lines.append(line.rstrip())

    base_name = os.path.splitext(os.path.basename(filename))[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if netlist_lines:
        with open(os.path.join(output_folder, f"{base_name}_netlist.rpt"), 'w') as nf:
            nf.writelines(netlist_lines)

    if complist_lines:
        with open(os.path.join(output_folder, f"{base_name}_complist.rpt"), 'w') as cf:
            cf.writelines(complist_lines)

    if fsm_lines:
        with open(os.path.join(output_folder, f"{base_name}_fsm.rpt"), 'w') as ff:
            ff.writelines(fsm_lines)

    if transitions:
        # Convert keys to int for JSON output
        transitions_int_keys = {int(k): v for k, v in transitions.items()}
        with open(os.path.join(output_folder, f"{base_name}_state_transitions.json"), 'w') as tf:
            json.dump(transitions_int_keys, tf, indent=2)

    if stg_lines:
        stg_out_path = os.path.join(output_folder, f"{base_name}_stg.rpt")
        with open(stg_out_path, 'w') as out_f:
            for line in stg_lines:
                out_f.write(line + "\n")

    # last, copy the original file to the output folder
    with open(filename, 'r') as f:
        original_content = f.read()
    original_file_path = os.path.join(output_folder, f"{base_name}.rpt")
    with open(original_file_path, 'w') as of:
        of.write(original_content)
    print(f"Copied original file to {original_file_path}")


def extract_all_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('verbose.rpt'):
            full_path = os.path.join(input_folder, filename)
            subfolder_name = filename.replace('.verbose.rpt', '')
            subfolder_path = os.path.join(output_folder, subfolder_name)
            extract_sections(full_path, subfolder_path)

def parse_fsm_report_to_cdfg(report_lines):
    import json
    import re

    cdfg = {}
    current_state = None
    state_pattern = re.compile(r"State\s+(\d+)\s*<SV\s*=\s*\d+>\s*<Delay\s*=\s*[\d\.]+>")
    call_pattern = re.compile(
        r'"(?P<dest>%\w+)\s*=\s*call\s+void\s+@(?P<func>\w+),\s*(?P<args>.+)"'
    )
    other_pattern = re.compile(
        r'"(?P<dest>%\w+)\s*=\s*(?P<op>\w+)\s+(.+)"'
    )
    ret_pattern = re.compile(
        r'"(?P<dest>%\w+)\s*=\s*ret"'
    )
    delay_pattern = re.compile(
        r"<Delay\s*=\s*([\d\.]+)>"
    )
    predicate_pattern = re.compile(r"<Predicate\s*=\s*([^\>]+)>")
    fraction_pattern = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")

    # Pattern to extract type from alloca and read ops
    type_pattern = re.compile(r'alloca\s+(\w+)\s+\d+')
    read_pattern = re.compile(r'read\s+(\w+)\s+@[^,]+,\s*(\w+)\s+(%?\w+)')

    for line in report_lines:
        # Detect state header
        state_match = state_pattern.match(line)
        if state_match:
            current_state = int(state_match.group(1))
            if current_state not in cdfg:
                cdfg[current_state] = []
            continue

        # Extract the fraction if present
        fraction_match = fraction_pattern.search(line)
        steps_remaining = None
        total_steps = None
        if fraction_match:
            steps_remaining = int(fraction_match.group(1))
            total_steps = int(fraction_match.group(2))

        # Only process lines with at least two "-->"
        parts = line.split('-->')
        if len(parts) < 2 or current_state is None:
            continue
        ir = parts[1].strip()
        op_info = parts[2] if len(parts) > 2 else ""

        # Extract predicate if present
        predicate_match = predicate_pattern.search(op_info)
        predicate = predicate_match.group(1).strip() if predicate_match else None

        # Try to match a call
        m = call_pattern.search(ir)
        if m:
            dest = m.group('dest')
            func = m.group('func')
            args = m.group('args')
            sources = []
            for arg in args.split(','):
                arg = arg.strip()
                if ' ' in arg:
                    dtype, var = arg.split(' ', 1)
                    sources.append({"source": var, "type": dtype})
            delay_match = delay_pattern.search(op_info)
            delay = float(delay_match.group(1)) if delay_match else 0.0
            op_dict = {
                "operator": "call",
                "function": func,
                "sources": sources,
                "destination": dest,
                "delay": delay
            }
            if predicate is not None:
                op_dict["predicate"] = predicate
            if steps_remaining is not None and total_steps is not None:
                op_dict["steps_remaining"] = steps_remaining
                op_dict["total_steps"] = total_steps
            cdfg[current_state].append(op_dict)
            continue

        # Try to match ret operation
        m = ret_pattern.search(ir)
        if m:
            dest = m.group('dest')
            delay_match = delay_pattern.search(op_info)
            delay = float(delay_match.group(1)) if delay_match else 0.0
            op_dict = {
                "operator": "ret",
                "sources": [],
                "destination": dest,
                "delay": delay
            }
            if predicate is not None:
                op_dict["predicate"] = predicate
            if steps_remaining is not None and total_steps is not None:
                op_dict["steps_remaining"] = steps_remaining
                op_dict["total_steps"] = total_steps
            cdfg[current_state].append(op_dict)
            continue

        # Try to match other operations
        m = other_pattern.search(ir)
        if m:
            dest = m.group('dest')
            op = m.group('op')
            rest = m.group(3)
            sources = []
            # Special handling for alloca and read
            if op == "alloca":
                tpm = type_pattern.search(ir)
                dtype = tpm.group(1) if tpm else None
                src = rest.split()[-1] if rest.split() else ""
                sources = [{"source": src, "type": dtype}]
                op_dict = {
                    "operator": op,
                    "sources": sources,
                    "destination": dest,
                    "dest_data_type": dtype,
                    "delay": float(delay_pattern.search(op_info).group(1)) if delay_pattern.search(op_info) else 0.0
                }
            elif op == "read":
                rpm = read_pattern.search(ir)
                if rpm:
                    dtype = rpm.group(1)
                    src = rpm.group(3)
                    sources = [{"source": src, "type": dtype}]
                    op_dict = {
                        "operator": op,
                        "sources": sources,
                        "destination": dest,
                        "dest_data_type": dtype,
                        "delay": float(delay_pattern.search(op_info).group(1)) if delay_pattern.search(op_info) else 0.0
                    }
                else:
                    op_dict = {
                        "operator": op,
                        "sources": sources,
                        "destination": dest,
                        "delay": float(delay_pattern.search(op_info).group(1)) if delay_pattern.search(op_info) else 0.0
                    }
            else:
                for arg in rest.split(','):
                    arg = arg.strip()
                    if ' ' in arg:
                        dtype, var = arg.split(' ', 1)
                        sources.append({"source": var, "type": dtype})
                op_dict = {
                    "operator": op,
                    "sources": sources,
                    "destination": dest,
                    "delay": float(delay_pattern.search(op_info).group(1)) if delay_pattern.search(op_info) else 0.0
                }
            if predicate is not None:
                op_dict["predicate"] = predicate
            if steps_remaining is not None and total_steps is not None:
                op_dict["steps_remaining"] = steps_remaining
                op_dict["total_steps"] = total_steps
            cdfg[current_state].append(op_dict)

    # Ensure top-level keys are integers in the output JSON
    cdfg_int_keys = {int(k): v for k, v in cdfg.items()}
    return json.dumps(cdfg_int_keys, indent=2)

def parse_all_fsm_reports(output_folder):
    for subfolder in os.listdir(output_folder):
        subfolder_path = os.path.join(output_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for fname in os.listdir(subfolder_path):
                if fname.endswith('_fsm.rpt'):
                    fsm_path = os.path.join(subfolder_path, fname)
                    cdfg_path = os.path.join(subfolder_path, fname.replace('_fsm.rpt', '_fsm.json'))
                    with open(fsm_path, 'r') as f:
                        lines = f.readlines()
                    cdfg = parse_fsm_report_to_cdfg(lines)
                    with open(cdfg_path, 'w') as f:
                        f.write(cdfg)
                    # Parse and write stg report to JSON
                    stg_path = os.path.join(subfolder_path, fname.replace('_fsm.rpt', '_stg.rpt'))
                    if os.path.exists(stg_path):
                        parse_stg_rpt_to_json(stg_path)

def parse_stg_rpt_to_json(stg_rpt_path):

    base_name = os.path.basename(stg_rpt_path).replace('.verbose_stg.rpt', '')
    output_json = os.path.join(os.path.dirname(stg_rpt_path), base_name + '.verbose_STG_IN_OUT.json')

    with open(stg_rpt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    result = {
        "metadata": {},
        "inputs": [],
        "outputs": [],
        "completion": None
    }

    # Parse metadata lines at the start
    for line in lines:
        if line.startswith('- Is '):
            meta_match = re.match(r"- Is\s+(.+?):\s*(.+)", line)
            if meta_match:
                key = meta_match.group(1).strip()
                val = meta_match.group(2).strip()
                try:
                    val = int(val)
                except ValueError:
                    pass
                result["metadata"][key] = val
        elif line.startswith("Port ["):
            # Stop parsing metadata when ports start
            break

    # Now parse ports and completion
    for line in lines:
        # Parse Return port
        if line.startswith("Port [ Return ]"):
            port_info = {"name": "Return"}
            io_mode_match = re.search(r"IO mode=([^\s:;]+)", line)
            if io_mode_match:
                port_info["io_mode"] = io_mode_match.group(1)
            wired_match = re.search(r"wired[:=](\d+)", line)
            if wired_match:
                port_info["wired"] = int(wired_match.group(1))
            result["completion"] = port_info
            continue

        # Parse other ports
        port_match = re.match(r"Port \[ ([^\]]+)]:\s*(.*)", line)
        if port_match:
            port_name = port_match.group(1)
            rest = port_match.group(2)
            port_info = {"name": port_name}
            # Extract all flags and IO mode
            for key in ["wired", "compound", "hidden", "nouse", "global", "static", "extern", "dir", "type", "pingpong", "private_global"]:
                m = re.search(rf"{key}[:=](\d+)", rest)
                if m:
                    port_info[key] = int(m.group(1))
            io_mode_match = re.search(r"IO mode=([^\s:;]+)", rest)
            if io_mode_match:
                port_info["io_mode"] = io_mode_match.group(1)
            # Determine input/output direction
            # dir=0: input, dir=1: output
            if "dir" in port_info:
                if port_info["dir"] == 0:
                    result["inputs"].append(port_info)
                elif port_info["dir"] == 1:
                    result["outputs"].append(port_info)
            else:
                result["inputs"].append(port_info)

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

# Example usage after extract_all_files:
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python extract_sections.py <input_directory> [output_folder]")
    else:
        input_directory = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) == 3 else "."
        extract_all_files(input_directory, output_folder)
        parse_all_fsm_reports(output_folder)



