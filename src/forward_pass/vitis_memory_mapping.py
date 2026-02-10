"""
Cross-hierarchy memory resource mapping for Vitis HLS designs.

Parses generated Verilog to trace memory (ap_memory) and FIFO (ap_fifo)
port connections from sub-modules back to their parent-level RAM/FIFO
instances. Produces a JSON mapping that links each sub-module's local
variable names to the actual storage instances they connect to.

The generated Verilog uses named port binding in module instantiations,
which is the only reliable source for this cross-hierarchy mapping.
The HLS reports and XML metadata do not explicitly store it.

Usage:
    python vitis_memory_mapping.py <verilog_dir> <output_dir> [--top <module>]

Example:
    python vitis_memory_mapping.py solution1/impl/verilog/ parse_results/ --top forward
"""

import sys
import os
import re
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

DEBUG = True

def log_info(msg):
    if DEBUG:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Verilog parsing
# ---------------------------------------------------------------------------

def _read_verilog(path):
    with open(path, 'r') as f:
        return f.read()


def parse_module_instantiations(verilog_text):
    """
    Extract all module instantiations and their named port connections from
    a Verilog source file.

    Returns a list of dicts:
        [{ 'module_type': str,
           'instance_name': str,
           'ports': { port_name: signal_name, ... }
        }, ...]
    """
    instances = []

    # Strip Verilog parameter blocks: #( .Param(val), ... )
    # so that parameterised instantiations reduce to the simple
    # <module_type> <instance_name>( ... ); form.
    text = re.sub(
        r'\s*#\s*\((?:[^()]*\([^()]*\))*[^()]*\)',
        '', verilog_text, flags=re.DOTALL)

    # Match: <module_type> <instance_name>( ... );
    # The port block can span thousands of lines, so we use a non-greedy match
    # anchored by the closing ");".
    inst_re = re.compile(
        r'(\w+)\s+(\w+)\s*\(\s*\n(.*?)\);',
        re.DOTALL
    )
    port_re = re.compile(r'\.(\w+)\s*\(\s*([^)]*)\s*\)')

    for m in inst_re.finditer(text):
        module_type = m.group(1)
        instance_name = m.group(2)
        ports_text = m.group(3)

        ports = {}
        for pm in port_re.finditer(ports_text):
            port_name = pm.group(1)
            signal = pm.group(2).strip()
            if signal:
                ports[port_name] = signal

        instances.append({
            'module_type': module_type,
            'instance_name': instance_name,
            'ports': ports,
        })

    return instances


# ---------------------------------------------------------------------------
# Memory interface detection
# ---------------------------------------------------------------------------

# ap_memory port suffixes (dual-port read/write)
_MEM_SUFFIXES = [
    '_address0', '_ce0', '_q0', '_d0', '_we0',
    '_address1', '_ce1', '_q1', '_d1', '_we1',
]

# ap_fifo port suffixes
_FIFO_SUFFIXES = [
    '_din', '_full_n', '_write',
    '_dout', '_empty_n', '_read',
    '_num_data_valid', '_fifo_cap',
]


def _strip_suffix(name, suffixes):
    """Return (base, suffix) if *name* ends with one of *suffixes*, else None."""
    for sfx in suffixes:
        if name.endswith(sfx):
            return name[: -len(sfx)], sfx[1:]     # drop leading '_'
    return None


def extract_memory_interfaces(ports):
    """
    Group port connections into ap_memory interfaces.

    Returns { mem_name: { suffix: signal, ... } }
    Only interfaces that have at least address0+ce0 are kept.
    """
    groups = defaultdict(dict)
    for port_name, signal in ports.items():
        hit = _strip_suffix(port_name, _MEM_SUFFIXES)
        if hit:
            base, sfx = hit
            groups[base][sfx] = signal

    return {
        name: info
        for name, info in groups.items()
        if 'address0' in info and 'ce0' in info
    }


def extract_fifo_interfaces(ports):
    """
    Group port connections into ap_fifo interfaces.

    Returns { fifo_name: { suffix: signal, ... } }
    """
    groups = defaultdict(dict)
    for port_name, signal in ports.items():
        hit = _strip_suffix(port_name, _FIFO_SUFFIXES)
        if hit:
            base, sfx = hit
            groups[base][sfx] = signal

    return {
        name: info
        for name, info in groups.items()
        if ('din' in info and 'full_n' in info) or
           ('dout' in info and 'empty_n' in info)
    }


# ---------------------------------------------------------------------------
# Signal-to-resource name extraction
# ---------------------------------------------------------------------------

def ram_name_from_signal(signal):
    """
    Derive the parent RAM instance base name from an RTL signal connected
    to a q0/q1 read-data port.

    Patterns observed in Vitis HLS generated Verilog:
        v418_q0       -> v418
        v418_t_q0     -> v418
        v418_1_t_q0   -> v418_1
        v418_10_t_q0  -> v418_10

    Returns the base name string, or None if the signal does not match.
    """
    m = re.match(r'(.+?)(?:_t)?_q[01]$', signal)
    return m.group(1) if m else None


def fifo_name_from_signal(signal, suffix_type):
    """
    Derive the parent FIFO instance base name from an RTL signal.

    For write-side FIFOs the identifying signal is full_n:
        v417_full_n    -> v417
        v417_4_full_n  -> v417_4

    For read-side FIFOs the identifying signal is empty_n:
        v426_empty_n   -> v426
    """
    if suffix_type == 'full_n':
        m = re.match(r'(.+?)_full_n$', signal)
    elif suffix_type == 'empty_n':
        m = re.match(r'(.+?)_empty_n$', signal)
    else:
        return None
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Verbose-report Memory table parsing
# ---------------------------------------------------------------------------

def parse_memory_table(report_path):
    """
    Parse the '* Memory:' table from a verbose report to get RAM metadata.

    Returns { ram_name: { 'module': str, 'bram18k': int, 'ff': int,
              'lut': int, 'uram': int, 'words': int, 'bits': int,
              'banks': int, 'total_size': int } }
    """
    rams = {}
    in_memory = False
    header_seen = False

    with open(report_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped == '* Memory:':
                in_memory = True
                header_seen = False
                continue

            if in_memory:
                if stripped.startswith('+---'):
                    continue
                if not stripped.startswith('|'):
                    if header_seen:
                        break          # end of table
                    continue

                fields = [c.strip() for c in stripped.split('|')]
                # fields[0] and fields[-1] are empty from leading/trailing |
                if len(fields) < 10:
                    continue

                if fields[1] == 'Memory':
                    header_seen = True
                    continue

                name = fields[1]
                if name == 'Total':
                    continue
                try:
                    entry = {
                        'module': fields[2],
                        'bram18k': int(fields[3]),
                        'ff': int(fields[4]),
                        'lut': int(fields[5]),
                        'uram': int(fields[6]),
                        'words': int(fields[7]),
                        'bits': int(fields[8]),
                        'banks': int(fields[9]),
                    }
                    # W*Bits*Banks column
                    if len(fields) > 10 and fields[10]:
                        entry['total_size'] = int(fields[10])
                    else:
                        entry['total_size'] = (entry['words']
                                               * entry['bits']
                                               * entry['banks'])
                    rams[name] = entry
                except (ValueError, IndexError):
                    pass

    return rams


def parse_fifo_table(report_path):
    """
    Parse the '* FIFO:' table from a verbose report to get FIFO metadata.

    Returns { fifo_name: { 'bram18k': int, 'ff': int, 'lut': int,
              'uram': int, 'depth': int, 'bits': int,
              'total_size': int } }
    """
    fifos = {}
    in_fifo = False
    header_seen = False

    with open(report_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped == '* FIFO:':
                in_fifo = True
                header_seen = False
                continue

            if in_fifo:
                if stripped.startswith('+---'):
                    continue
                if not stripped.startswith('|'):
                    if header_seen:
                        break          # end of table
                    continue

                fields = [c.strip() for c in stripped.split('|')]
                if len(fields) < 8:
                    continue

                if fields[1] == 'Name':
                    header_seen = True
                    continue

                name = fields[1]
                if name == 'Total':
                    continue
                try:
                    uram_str = fields[5]
                    uram_val = 0 if uram_str == '-' else int(uram_str)
                    entry = {
                        'bram18k': int(fields[2]),
                        'ff': int(fields[3]),
                        'lut': int(fields[4]),
                        'uram': uram_val,
                        'depth': int(fields[6]),
                        'bits': int(fields[7]),
                    }
                    # Size:D*B column
                    if len(fields) > 8 and fields[8]:
                        entry['total_size'] = int(fields[8])
                    else:
                        entry['total_size'] = entry['depth'] * entry['bits']
                    fifos[name] = entry
                except (ValueError, IndexError):
                    pass

    return fifos


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def _build_signal_to_ram_map(all_instances):
    """
    Build a reverse map from signals connected to RAM instances back to
    the RAM base name.

    RAM instances end with '_U' (e.g. v429_U -> base name v429) and have
    address/ce ports (possibly prefixed with i_/t_ in Vitis HLS wrappers).

    Returns { signal: ram_base_name }
    """
    signal_to_ram = {}
    for inst in all_instances:
        instance_name = inst['instance_name']
        if not instance_name.endswith('_U'):
            continue
        ports = inst['ports']
        # RAM instances have address0+ce0 ports, possibly with i_/t_ prefix
        has_addr = ('address0' in ports or 'i_address0' in ports
                    or 't_address0' in ports)
        has_ce = ('ce0' in ports or 'i_ce0' in ports
                  or 't_ce0' in ports)
        if not (has_addr and has_ce):
            continue

        ram_base = instance_name[:-2]       # strip '_U'
        log_info(f"  RAM instance {instance_name} -> base {ram_base}")
        for signal in ports.values():
            if signal:
                signal_to_ram[signal] = ram_base

    log_info(f"Signal-to-RAM map: {len(signal_to_ram)} entries")
    return signal_to_ram


def _child_module_name(instance_name):
    """node4_U0 -> node4"""
    if instance_name.endswith('_U0'):
        return instance_name[:-3]
    return instance_name


def build_memory_mapping(verilog_dir, top_module='forward', report_dir=None):
    """
    Build a cross-hierarchy memory mapping for a Vitis HLS design.

    Parameters
    ----------
    verilog_dir : str
        Path to the generated Verilog directory (e.g. solution1/impl/verilog/).
    top_module : str
        Name of the top-level module (default 'forward').
    report_dir : str or None
        Optional path to the parsed report directory.  When given, the
        mapping is enriched with RAM metadata (width, depth, BRAM usage)
        from each module's verbose report.

    Returns
    -------
    dict  –  Structured mapping, keyed by child module name.
    """
    top_v_path = os.path.join(verilog_dir, f'{top_module}.v')
    if not os.path.isfile(top_v_path):
        raise FileNotFoundError(f'Top-level Verilog not found: {top_v_path}')

    log_info(f"Parsing top-level Verilog: {top_v_path}")
    top_text = _read_verilog(top_v_path)
    all_instances = parse_module_instantiations(top_text)
    log_info(f"Found {len(all_instances)} module instantiations")

    # Optionally load top-module RAM/FIFO metadata from verbose report
    top_ram_meta = {}
    top_fifo_meta = {}
    if report_dir:
        top_rpt = os.path.join(report_dir, top_module,
                               f'{top_module}.verbose.rpt')
        if not os.path.isfile(top_rpt):
            top_rpt = os.path.join(report_dir, top_module,
                                   f'{top_module}.rpt')
        if os.path.isfile(top_rpt):
            top_ram_meta = parse_memory_table(top_rpt)
            top_fifo_meta = parse_fifo_table(top_rpt)
            log_info(f"Loaded {len(top_ram_meta)} RAM + "
                     f"{len(top_fifo_meta)} FIFO metadata entries "
                     f"from {top_rpt}")

    # Build reverse map: signal -> RAM base name (for write-only port resolution)
    signal_to_ram = _build_signal_to_ram_map(all_instances)

    # Set of all instance names – used to distinguish internal RAMs from
    # top-level interface ports (interface ports have no _U instance).
    instance_names = {inst['instance_name'] for inst in all_instances}

    mapping = {}

    for inst in all_instances:
        instance_name = inst['instance_name']
        child_module = _child_module_name(instance_name)
        ports = inst['ports']

        # --- ap_memory interfaces -------------------------------------------
        mem_ifs = extract_memory_interfaces(ports)
        if mem_ifs:
            log_info(f"Processing {child_module} ({instance_name}): "
                     f"{len(mem_ifs)} memory interface(s)")
        mem_entries = {}
        for mem_name, port_info in mem_ifs.items():
            entry = {
                'child_port': mem_name,
                'port_type': 'ap_memory',
            }

            # Determine read / write / read-write
            has_read = 'q0' in port_info or 'q1' in port_info
            has_write = 'd0' in port_info or 'we0' in port_info
            if has_read and has_write:
                entry['direction'] = 'read_write'
            elif has_write:
                entry['direction'] = 'write'
            else:
                entry['direction'] = 'read'

            # Trace back to parent RAM via q0 signal
            parent_ram = None
            for q_port in ('q0', 'q1'):
                if q_port in port_info:
                    parent_ram = ram_name_from_signal(port_info[q_port])
                    if parent_ram:
                        log_info(f"  {child_module}.{mem_name}: resolved via "
                                 f"{q_port} signal '{port_info[q_port]}' -> "
                                 f"parent RAM {parent_ram}")
                        break

            # Fallback: match write-side signals against RAM instantiations
            if parent_ram is None:
                log_info(f"  {child_module}.{mem_name}: no q0/q1 signal, "
                         f"trying write-side fallback "
                         f"(direction={entry['direction']})")
                for sig_port in ('d0', 'we0', 'address0', 'ce0',
                                 'd1', 'we1', 'address1', 'ce1'):
                    if sig_port in port_info:
                        sig = port_info[sig_port]
                        if sig in signal_to_ram:
                            parent_ram = signal_to_ram[sig]
                            log_info(f"  {child_module}.{mem_name}: resolved "
                                     f"via {sig_port} signal '{sig}' -> "
                                     f"parent RAM {parent_ram}")
                            break
                        else:
                            log_info(f"  {child_module}.{mem_name}: {sig_port} "
                                     f"signal '{sig}' not in RAM map")

            # Classify: internal_ram vs top_interface
            if parent_ram is not None:
                entry['parent_ram'] = parent_ram
                entry['parent_module'] = top_module
                ram_inst_name = parent_ram + '_U'

                if ram_inst_name in instance_names:
                    entry['storage_type'] = 'internal_ram'
                    # Attach RAM metadata if available
                    if ram_inst_name in top_ram_meta:
                        meta = top_ram_meta[ram_inst_name]
                        entry['ram_module'] = meta['module']
                        entry['width'] = meta['bits']
                        entry['depth'] = meta['words']
                        entry['bram18k'] = meta['bram18k']
                        entry['ff'] = meta['ff']
                        entry['lut'] = meta['lut']
                        entry['uram'] = meta['uram']
                        entry['total_size'] = meta['total_size']
                        log_info(f"  {child_module}.{mem_name}: internal_ram "
                                 f"-> {parent_ram} ({meta['module']}, "
                                 f"{meta['words']}x{meta['bits']}, "
                                 f"total_size={meta['total_size']})")
                    else:
                        log_info(f"  {child_module}.{mem_name}: internal_ram "
                                 f"-> {parent_ram} (no metadata for "
                                 f"'{ram_inst_name}')")
                else:
                    entry['storage_type'] = 'top_interface'
                    log_info(f"  {child_module}.{mem_name}: top_interface "
                             f"-> {parent_ram} (no '{ram_inst_name}' "
                             f"instance found)")
            else:
                # Unresolved — after RAM-map fallback the only remaining
                # unresolved write-only ports are top-level interface signals.
                entry['storage_type'] = 'top_interface'
                log_info(f"  {child_module}.{mem_name}: top_interface "
                         f"(unresolved, likely top-level I/O)")

            mem_entries[mem_name] = entry

        # --- ap_fifo interfaces ---------------------------------------------
        fifo_ifs = extract_fifo_interfaces(ports)
        if fifo_ifs:
            log_info(f"Processing {child_module} ({instance_name}): "
                     f"{len(fifo_ifs)} FIFO interface(s)")
        fifo_entries = {}
        for fifo_name, port_info in fifo_ifs.items():
            entry = {
                'child_port': fifo_name,
                'port_type': 'ap_fifo',
            }

            if 'din' in port_info:
                entry['direction'] = 'write'
                parent_fifo = fifo_name_from_signal(
                    port_info.get('full_n', ''), 'full_n')
            else:
                entry['direction'] = 'read'
                parent_fifo = fifo_name_from_signal(
                    port_info.get('empty_n', ''), 'empty_n')

            if parent_fifo is not None:
                entry['parent_fifo'] = parent_fifo
                entry['parent_module'] = top_module

                # Attach FIFO metadata if available
                fifo_inst_name = parent_fifo + '_U'
                if fifo_inst_name in top_fifo_meta:
                    meta = top_fifo_meta[fifo_inst_name]
                    entry['depth'] = meta['depth']
                    entry['width'] = meta['bits']
                    entry['bram18k'] = meta['bram18k']
                    entry['ff'] = meta['ff']
                    entry['lut'] = meta['lut']
                    entry['uram'] = meta['uram']
                    entry['total_size'] = meta['total_size']
                    log_info(f"  {child_module}.{fifo_name}: -> parent FIFO "
                             f"{parent_fifo} (depth={meta['depth']}, "
                             f"bits={meta['bits']}, "
                             f"total_size={meta['total_size']})")
                else:
                    log_info(f"  {child_module}.{fifo_name}: -> parent FIFO "
                             f"{parent_fifo} (no metadata for "
                             f"'{fifo_inst_name}')")
            else:
                log_info(f"  {child_module}.{fifo_name}: WARNING - could not "
                         f"resolve parent FIFO")

            fifo_entries[fifo_name] = entry

        if mem_entries or fifo_entries:
            mapping[child_module] = {}
            if mem_entries:
                mapping[child_module]['memory_ports'] = mem_entries
            if fifo_entries:
                mapping[child_module]['fifo_ports'] = fifo_entries

    # --- local memories inside each child module ----------------------------
    # Parse each child module's verbose report for RAM/FIFO instances that
    # are local to the submodule (not connected through the parent).
    if report_dir:
        for child_module, child_info in mapping.items():
            child_rpt = os.path.join(report_dir, child_module,
                                     f'{child_module}.verbose.rpt')
            if not os.path.isfile(child_rpt):
                child_rpt = os.path.join(report_dir, child_module,
                                         f'{child_module}.rpt')
            if not os.path.isfile(child_rpt):
                continue

            child_ram_meta = parse_memory_table(child_rpt)
            child_fifo_meta = parse_fifo_table(child_rpt)

            if not child_ram_meta and not child_fifo_meta:
                continue

            log_info(f"Local memories for {child_module}: "
                     f"{len(child_ram_meta)} RAM, "
                     f"{len(child_fifo_meta)} FIFO")

            mem_ports = child_info.setdefault('memory_ports', {})
            for ram_name, meta in child_ram_meta.items():
                ram_base = (ram_name[:-2]
                            if ram_name.endswith('_U') else ram_name)
                if ram_base in mem_ports:
                    continue        # already captured via parent connection
                entry = {
                    'child_port': ram_base,
                    'port_type': 'ap_memory',
                    'direction': 'read_write',
                    'storage_type': 'internal_ram',
                    'parent_ram': ram_base,
                    'parent_module': child_module,
                    'ram_module': meta['module'],
                    'width': meta['bits'],
                    'depth': meta['words'],
                    'bram18k': meta['bram18k'],
                    'ff': meta['ff'],
                    'lut': meta['lut'],
                    'uram': meta['uram'],
                    'total_size': meta['total_size'],
                }
                mem_ports[ram_base] = entry
                log_info(f"  {child_module}.{ram_base}: local RAM "
                         f"({meta['module']}, "
                         f"{meta['words']}x{meta['bits']}, "
                         f"total_size={meta['total_size']})")

            fifo_ports = child_info.setdefault('fifo_ports', {})
            for fifo_name, meta in child_fifo_meta.items():
                fifo_base = (fifo_name[:-2]
                             if fifo_name.endswith('_U') else fifo_name)
                if fifo_base in fifo_ports:
                    continue        # already captured via parent connection
                entry = {
                    'child_port': fifo_base,
                    'port_type': 'ap_fifo',
                    'direction': 'read_write',
                    'storage_type': 'internal_fifo',
                    'parent_fifo': fifo_base,
                    'parent_module': child_module,
                    'depth': meta['depth'],
                    'width': meta['bits'],
                    'bram18k': meta['bram18k'],
                    'ff': meta['ff'],
                    'lut': meta['lut'],
                    'uram': meta['uram'],
                    'total_size': meta['total_size'],
                }
                fifo_ports[fifo_base] = entry
                log_info(f"  {child_module}.{fifo_base}: local FIFO "
                         f"(depth={meta['depth']}, bits={meta['bits']}, "
                         f"total_size={meta['total_size']})")

    log_info(f"Memory mapping complete: {len(mapping)} child module(s) mapped")
    return mapping


def build_memory_mapping_recursive(verilog_dir, top_module='forward',
                                   report_dir=None):
    """
    Recursively build memory mappings through the full module hierarchy.

    Starts from *top_module* and follows each sub-module that has its own
    Verilog file, producing a nested mapping where every memory reference
    at any level can be traced back to the originating RAM instance.

    Returns
    -------
    dict  –  { child_module: { 'memory_ports': {...}, 'fifo_ports': {...},
               'sub_mappings': { grandchild: ... } } }
    """
    # First level
    mapping = build_memory_mapping(verilog_dir, top_module, report_dir)

    # For each child, check if it has its own sub-modules
    for child_module, child_info in mapping.items():
        child_v_name = f'{top_module}_{child_module}.v'
        child_v_path = os.path.join(verilog_dir, child_v_name)
        if os.path.isfile(child_v_path):
            sub_mapping = build_memory_mapping(
                verilog_dir,
                top_module=f'{top_module}_{child_module}',
                report_dir=report_dir)
            if sub_mapping:
                child_info['sub_mappings'] = sub_mapping

    return mapping


# ---------------------------------------------------------------------------
# Flattened lookup helpers
# ---------------------------------------------------------------------------

def flatten_memory_mapping(mapping, prefix=''):
    """
    Flatten the hierarchical mapping into a simple lookup table.

    Returns a list of dicts, each with keys:
        module, local_name, parent_module, parent_ram, width, depth, ...
    """
    rows = []
    for child_module, info in mapping.items():
        mod_path = f'{prefix}/{child_module}' if prefix else child_module

        for port_name, entry in info.get('memory_ports', {}).items():
            row = {'module': mod_path, 'local_name': port_name}
            row.update(entry)
            rows.append(row)

        for port_name, entry in info.get('fifo_ports', {}).items():
            row = {'module': mod_path, 'local_name': port_name}
            row.update(entry)
            rows.append(row)

        if 'sub_mappings' in info:
            rows.extend(
                flatten_memory_mapping(info['sub_mappings'], prefix=mod_path))

    return rows


def lookup(mapping, module, local_name):
    """
    Look up a single variable reference.

    Parameters
    ----------
    mapping : dict   – output of build_memory_mapping()
    module : str     – child module name  (e.g. 'node4')
    local_name : str – local variable name (e.g. 'v61_0_0_0')

    Returns the entry dict with parent_ram / parent_fifo info, or None.
    """
    mod_info = mapping.get(module, {})
    for section in ('memory_ports', 'fifo_ports'):
        entry = mod_info.get(section, {}).get(local_name)
        if entry:
            return entry
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_mapping(mapping, output_dir, filename='memory_mapping.json'):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    return out_path

def load_mapping(output_dir, filename='memory_mapping.json'):
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'r') as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Map cross-hierarchy memory resources in Vitis HLS designs')
    parser.add_argument('verilog_dir',
                        help='Path to generated Verilog directory')
    parser.add_argument('output_dir',
                        help='Path to write memory_mapping.json')
    parser.add_argument('--top', default='forward',
                        help='Top-level module name (default: forward)')
    parser.add_argument('--reports', default=None,
                        help='Path to parsed report directory for RAM metadata')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively map through sub-module hierarchies')
    parser.add_argument('--flat', action='store_true',
                        help='Also write a flattened CSV-style lookup table')

    args = parser.parse_args()

    if args.recursive:
        mapping = build_memory_mapping_recursive(
            args.verilog_dir, args.top, args.reports)
    else:
        mapping = build_memory_mapping(
            args.verilog_dir, args.top, args.reports)

    out_path = write_mapping(mapping, args.output_dir)
    print(f'Wrote {out_path}')

    if args.flat:
        flat = flatten_memory_mapping(mapping)
        flat_path = os.path.join(args.output_dir, 'memory_mapping_flat.json')
        with open(flat_path, 'w') as f:
            json.dump(flat, f, indent=2)
        print(f'Wrote {flat_path}')


if __name__ == '__main__':
    main()
