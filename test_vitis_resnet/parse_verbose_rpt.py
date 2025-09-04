import sys
import os
import re
import json

# Regexes
_LOOP_SECTION_RX = re.compile(r'^\s*\*\s*Loop:\s*$', re.IGNORECASE)
_HEADER_RX       = re.compile(r'\|\s*Loop Name\s*\|', re.IGNORECASE)
_SEP_RX          = re.compile(r'^\s*-+\s*(\|\s*-+\s*)+$')       # -----|-----|---
_DATA_ROW_RX     = re.compile(r'^\s*\|\s*(?:-\s*)?(?P<name>[^|]+?)\s*\|')  # first cell only


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

    # Required variables for the loops.
    has_loop = False
    trip_counts : dict[str, int] = {}
    within_loop = False
    loop_count = 0
    loops = {}

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

            # Checking for the loops.
            # And if the loop is found then storing the trip count.
            if(within_loop):
                within_loop = process_loop_line(line, trip_counts, loop_count)
                loop_count = loop_count + 1
            if(not within_loop):
                within_loop = check_loop(line)
                has_loop = has_loop or within_loop
        for line in fsm_lines:
            loops1 = populate_loop(line, loops, trip_counts)
            if(loops1):
                loops[loops1["id"]] = loops1

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

    if loops:
        for props in loops.values():
            props["body_states"]  = list(props["body_states"])
            props["latch_states"] = list(props["latch_states"])

        #  Dumping the loop into a _loop.json
        with open(os.path.join(output_folder, f"{base_name}_loops_state.json"), 'w') as tf:
            json.dump(loops, tf, indent=2)
            if hasattr(populate_loop, "_ctx"):
                delattr(populate_loop, "_ctx")

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

def check_loop(line : str) -> bool:
    if 'Loop Name' in line:
        return True
    return False

def process_loop_line(line: str, trip_counts: dict, loop_count) -> bool:
# Process one line of a Vitis .rpt file.
# If the line contains a VITIS_LOOP row, extract loop_name and trip_count.
# Updates trip_counts in place and returns has_loop (True/False).
    # Checking if the loop_name is present in the line.
    if '+----------' in line:
        if(loop_count == 0 ):
            return True
        return False
    else:
        cols = [c.strip() for c in line.split("|") if c.strip()]

        if len(cols) < 7:
            return False  # not enough columns to safely parse

        loop_name = cols[0].lstrip("-").strip()
        try:
            trip_count = int(cols[6])   # 7th column (index 6)
        except ValueError:
            trip_count = 0

        trip_counts[loop_name] = trip_count
        return True
    return False



def print_loops_summary(has_loop: bool, trip_counts: dict):
    """
    Nicely print loops and their trip counts.
    """
    print("\n================ LOOP ANALYSIS ================")
    if not has_loop or not trip_counts:
        print("❌ No loops detected in this report.")
    else:
        print(f"✅ Loops detected: {len(trip_counts)}")
        for name, tc in trip_counts.items():
            print(f"   🔹 {name:<25} → Trip Count: {tc}")
    print("===============================================\n")

# ---------- regex helpers ----------
_RX_STATE_HDR = re.compile(r'^\s*State\s+(\d+)\b', re.IGNORECASE)      # "State 2 <SV=...>"
_RX_ST_LINE   = re.compile(r'\bST[_\s]*(\d+)\s*:', re.IGNORECASE)       # "ST_2 : Operation ..."
_RX_BR_I1     = re.compile(r'\bbr\s+i1\b.*?,\s*void\s*(%[A-Za-z0-9._]+)\s*,\s*void\s*(%[A-Za-z0-9._]+)')
# Back-edge to loop header label; tolerates "br void %for.cond..." / "br label %for.cond..."
_RX_BR_BACK   = re.compile(r'\bbr\b[^%]*(%for\.cond[0-9A-Za-z._]*)', re.IGNORECASE)
_RX_TRIPNUM   = re.compile(r'(?:i64|i32)\s+(\d+)')

def _ensure_entry(loops, loop_id) -> dict:
    """Create default loop entry if missing and return it."""
    if(loop_id not in loops):
        loops[loop_id] = {
            "id": loop_id,
            "target": None,
            "trip_count": 0,
            "header_label": None,
            "body_label": None,
            "exit_label": None,
            "backedge": False,
            "header_state": None,    # 'ST_<n>'
            "body_states": set(),
            "latch_states": set(),
        }
    return loops[loop_id]

def _state_name_from_line(line):
    """Return 'ST_<n>' if the line defines/mentions a state; else None."""
    m = _RX_STATE_HDR.match(line)
    if m:
        return f"ST_{int(m.group(1))}"
    m = _RX_ST_LINE.search(line)  # search because 'ST_2 :' can be later in line
    if m:
        return f"ST_{int(m.group(1))}"
    return None

def populate_loop(line, loops, known_loops):
    """
    Streaming parser: feed one line at a time.
    - Uses known loop IDs to associate info.
    - Tracks states as 'ST_<n>'.
    - Correctly fills latch_states even when the back-edge line doesn't mention the loop.

    Returns the loop entry updated by this line ({} if nothing updated).
    """
    s = line.rstrip("\n")

    # one shared sticky context across calls
    if not hasattr(populate_loop, "_ctx"):
        populate_loop._ctx = {
            "current_state": None,      # 'ST_<n>'
            "current_loop": None,       # loop id string
            "pending_trip": None,       # trip count seen before loop id
            "latch_by_header": {},      # header_label -> set('ST_<n>')
        }
    ctx = populate_loop._ctx
    updated = None

    # 1) track/normalize current state
    st = _state_name_from_line(s)
    if st:
        ctx["current_state"] = st

    # 2) if this line mentions any known loop id, select it as current loop & enrich
    hit_loop = None
    for lid in known_loops.keys():
        if lid in s:
            hit_loop = lid
            break
    if hit_loop:
        ent = _ensure_entry(loops, hit_loop)

        # best-effort target: token after '@' if it also contains the loop id
        at = s.find('@')
        if at != -1 and s.find(hit_loop, at) != -1:
            token = s[at+1:].split()[0].rstrip(',(')
            if token:
                ent["target"] = token

        # attach any earlier tripcount
        if ctx.get("pending_trip") is not None:
            ent["trip_count"] = known_loops[hit_loop]
            ctx["pending_trip"] = None

        # record body state
        if ctx.get("current_state"):
            ent["body_states"].add(ctx["current_state"])

        # if this loop already has a header_label, merge cached latch states
        hdr = ent.get("header_label")
        if hdr and hdr in ctx["latch_by_header"]:
            for st_name in ctx["latch_by_header"][hdr]:
                ent["latch_states"].add(st_name)
                ent["backedge"] = True

        ctx["current_loop"] = hit_loop
        updated = ent

    # 3) trip count: use current loop if known; else stash
    if "speclooptripcount" in s:
        nums = _RX_TRIPNUM.findall(s)
        tc = int(nums[-1]) if nums else 0
        if ctx.get("current_loop"):
            ent = _ensure_entry(loops, ctx["current_loop"])
            ent["trip_count"] = tc
            updated = ent
        else:
            ctx["pending_trip"] = tc
    # 4) header test: conditional branch with two successors → body/exit + header_state
    m_hdr = _RX_BR_I1.search(s)
    if m_hdr and ctx.get("current_loop"):
        ent = _ensure_entry(loops, ctx["current_loop"])
        if ent["body_label"] is None:
            ent["body_label"] = m_hdr.group(1)
        if ent["exit_label"] is None:
            ent["exit_label"] = m_hdr.group(2)
        if ent["header_state"] is None and ctx.get("current_state"):
            ent["header_state"] = ctx["current_state"]

        # If we already know header_label from a prior back-edge, merge cached latches now
        hdr = ent.get("header_label")
        if hdr and hdr in ctx["latch_by_header"]:
            for st_name in ctx["latch_by_header"][hdr]:
                ent["latch_states"].add(st_name)
                ent["backedge"] = True

        updated = ent
    # 5) latch/back-edge: check with `in` instead of regex
    if "br void %for.cond.i.i" in s or "br label %for.cond.i.i" in s:
        # extract the label (take the last token starting with %for.cond)
        header_label = None
        for token in s.split():
            if token.startswith("%for.cond"):
                header_label = token
        st_name = ctx.get("current_state")
        if header_label and st_name:
            ctx["latch_by_header"].setdefault(header_label, set()).add(st_name)

        if ctx.get("current_loop"):
            ent = _ensure_entry(loops, ctx["current_loop"])
            if ent["header_label"] is None:
                ent["header_label"] = header_label
            if st_name:
                ent["latch_states"].add(st_name)
            ent["backedge"] = True
            updated = ent
        else:
            for ent in loops.values():
                if ent.get("header_label") == header_label:
                    if st_name:
                        ent["latch_states"].add(st_name)
                    ent["backedge"] = True
                    updated = ent
    return updated


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
    # Updated call_pattern to match both void and non-void return types
    call_pattern = re.compile(
        r'"(?P<dest>%\w+)\s*=\s*call\s+(?P<rettype>\w+)\s+@(?P<func>\w+),\s*(?P<args>.+)"'
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

        # Try to match a call (void or non-void)
        m = call_pattern.search(ir)
        if m:
            dest = m.group('dest')
            rettype = m.group('rettype')
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
                "return_type": rettype,
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



