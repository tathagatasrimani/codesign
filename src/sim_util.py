import logging
import os
import glob
import datetime
from collections import defaultdict
import math
from sympy import Abs, exp, cosh, coth
import sympy as sp
import copy
import cvxpy as cp

logger = logging.getLogger(__name__)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import json
import pandas as pd
from pathlib import Path

def symbolic_convex_max(a, b, evaluate=True):
    if not isinstance(a, sp.Expr) and not isinstance(b, sp.Expr):
        return cp.maximum(a, b)
    """
    Max(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b + Abs(a - b, evaluate=evaluate))

def symbolic_min(a, b, evaluate=True):
    if not isinstance(a, sp.Expr) and not isinstance(b, sp.Expr):
        return cp.minimum(a, b)
    """
    Min(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b - Abs(a - b, evaluate=evaluate))

def custom_exp(x, evaluate=True):
    """
    Custom exp function to guard against overflow.
    """
    return exp(symbolic_min(500, x))

def custom_cosh(x, evaluate=True):
    """
    Custom cosh function to guard against overflow.
    """
    return cosh(symbolic_min(500, x))

def custom_coth(x, evaluate=True):
    """
    Custom coth function to guard against overflow.
    Also pyomo cannot handle coth, so use this function.
    """
    return (custom_exp(x) + custom_exp(-x)) / (custom_exp(x) - custom_exp(-x))

def custom_sech(x, evaluate=True):
    """
    Custom sech function to guard against overflow.
    """
    return 1 / custom_cosh(x)

def custom_pow(x, y, evaluate=True):
    """
    Custom pow function to guard against illegal negative base.
    """
    return pow(Abs(x, evaluate=evaluate), y)

# overwrite values of dict1 with values of dict2
# if a key is not present in dict1, still takes values from dict2
# if key is not present in dict2, keeps value from dict1
def deep_merge(dict1, dict2):
    result = dict(dict1)
    for key, value in dict2.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result.get(key, {}), value)
        else:
            result[key] = value
    return result

# merge model config up hierarchy to base cfg
def recursive_cfg_merge(model_cfgs, model_cfg_name):
    base_cfg = model_cfgs[model_cfg_name]["base_cfg"]
    model_cfg = model_cfgs[model_cfg_name]
    while True:
        model_cfg = deep_merge(model_cfgs[base_cfg], model_cfg)
        if base_cfg == "default":
            break
        base_cfg = model_cfgs[base_cfg]["base_cfg"]
    return model_cfg

def get_module_map():
    module_map = {
        # from HLS IR
        #"add": "Add16",
        "fadd": "Add16",
        "dadd": "Add16",
        #"sub": "Sub16",
        "fsub": "Sub16",
        "dsub": "Sub16",
        "dmul": "Mult16",
        #"mul": "Mult16",
        "fmul": "Mult16",
        #"div": "FloorDiv16",
        "fdiv": "FloorDiv16",
        "ddiv": "FloorDiv16",
        "lshr": "RShift16",
        "shl": "LShift16",
        "call": "Call",

        # from MLIR (blackboxed arith ops)
        "addf": "Add16",
        "subf": "Sub16",
        "mulf": "Mult16",
        "divf": "FloorDiv16",
        "exp_bb": "Exp16",
        "addf_ctrl_chain": "Add16",
        "subf_ctrl_chain": "Sub16",
        "mulf_ctrl_chain": "Mult16",
        "divf_ctrl_chain": "FloorDiv16",
        "exp_bb_ctrl_chain": "Exp16",
    }
    return module_map

def map_operator_types(full_netlist):
    """
    Map the operator types in the netlist to standardized function names using module_map.
    """
    module_map = get_module_map()
    for node in full_netlist:
        raw_fn = full_netlist.nodes[node].get('bind', {}).get('fcode')
        if raw_fn is None:
            raw_fn = "N/A"
        full_netlist.nodes[node]['function'] = module_map[raw_fn] if raw_fn in module_map else raw_fn
    return full_netlist

def get_latest_log_dir():
    log_dirs = glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), "../logs/*-*-*_*-*-*_*")))
    log_dirs = sorted(
        log_dirs,
        key=lambda x: datetime.datetime.strptime(x.split("/")[-1][:19], "%Y-%m-%d_%H-%M-%S"),
    )
    return log_dirs[-1]

def get_latest_log_dir_streamhls(path):
    log_dirs = glob.glob(os.path.normpath(os.path.join(path, "streamhls_*-*-*_*-*-*.log")))
    log_dirs = sorted(
        log_dirs,
        key=lambda x: datetime.datetime.strptime(x.split("/")[-1][10:29], "%Y-%m-%d_%H-%M-%S"),
    )
    return log_dirs[-1]

def change_clk_period_in_script(filename, new_period, hls_tool):
    CATAPULT_PERIOD_POSITION = -1
    new_lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line= line
            if hls_tool == "catapult":
                if line.find("set clk_period") != -1:
                    new_line = line.replace(line.split()[CATAPULT_PERIOD_POSITION], str(new_period))
            elif hls_tool == "vitis":
                if line.find("create_clock") != -1:
                    period_pos = line.split().index("-period")
                    new_line = line.replace(line.split()[period_pos+1], str(new_period))
            else:
                raise ValueError(f"Invalid hls tool: {hls_tool}")
            new_lines.append(new_line)
    with open(filename, "w") as f:
        f.writelines(new_lines)

# if expr is int or float, running xreplace will cause error. So add this safeguard. also supports cvxpy expressions.
def xreplace_safe(expr, replacements):
    if hasattr(expr, "value"):
        val = expr.value
        # Check if value is a numpy array - if so, extract scalar with .item()
        if isinstance(val, np.ndarray):
            return float(val.item())
        # Check if value is a numpy scalar type
        elif isinstance(val, (np.floating, np.integer)):
            return float(val)
        # Otherwise, return as-is (should be Python float/int)
        else:
            return val
    if not isinstance(expr, float) and not isinstance(expr, int):
        ret = expr.xreplace(replacements)
        if not isinstance(ret, float) and not isinstance(ret, int):
            assert not isinstance(ret, sp.Symbol), f"xreplace did not work, returned {ret}"
            return float(ret.evalf())
        else:
            return ret
    else:
        return expr

def add_area_constraint_to_script(filename, area_constraint):
    new_lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line = line
            if line.find("go architect") != -1:
                new_lines.append(new_line)
                new_line = f"directive set -AREA_GOAL {area_constraint}\n"
            new_lines.append(new_line)
    with open(filename, "w") as f:
        f.writelines(new_lines)

def remove_node(G, node):
    for src in G.predecessors(node):
        for dst in G.successors(node):
            G.add_edge(src, dst, weight=G.edges[src, node]["weight"], resource_edge=0)
    G.remove_node(node)

def filter_graph_by_function(graph, allowed_functions, exception_node_types=None):
    filtered_graph = copy.deepcopy(graph)
    for node in graph.nodes():
        if graph.nodes[node]['function'] not in allowed_functions and (exception_node_types is None or graph.nodes[node]['node_type'] not in exception_node_types):
            remove_node(filtered_graph, node)
    return filtered_graph

def topological_layout_plot(graph, filename, reverse=False, extra_edges=None):
    # Compute the topological order of the nodes
    if nx.is_directed_acyclic_graph(graph):
        topological_order = list(nx.topological_sort(graph))
    else:
        cycle = nx.find_cycle(graph)
        raise ValueError(f"Graph is not a Directed Acyclic Graph (DAG), topological sorting is not possible. Cycle is {cycle}")
    
    # Group nodes by level in topological order
    levels = defaultdict(int)
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}
    
    for node in topological_order:
        level = 0 if in_degrees[node] == 0 else max(levels[parent] + 1 for parent in graph.predecessors(node))
        levels[node] = level
    
    # Arrange nodes in horizontal groups based on level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    # Assign positions: group nodes by levels from top to bottom
    pos = {}
    for level, nodes in level_nodes.items():
        x_positions = np.linspace(-len(nodes)/2, len(nodes)/2, num=len(nodes))
        for x, node in zip(x_positions, nodes):
            pos[node] = (x, -level)

    if extra_edges:
        edge_colors = ['red' if (u, v) in extra_edges else 'gray' for (u, v) in graph.edges()]
    else:
        edge_colors = ['gray' for (u, v) in graph.edges()]
    
    # Draw the graph with curved edges to avoid overlap
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=700, font_size=10, connectionstyle="arc3,rad=0.2")
    
    # Draw dashed lines between topological levels
    max_level = max(level_nodes.keys())
    for level in range(max_level):
        plt.axhline(y=-(level + 0.5), color='gray', linestyle='dashed', linewidth=0.5)

    # Show the graph
    plt.savefig(filename, format='svg')
    plt.close()

def svg_plot(G, filename, extra_edges=None):

    # Create custom labels using the 'function' attribute if available
    labels = {}
    for node, data in G.nodes(data=True):
        if 'function' in data:
            labels[node] = f"{node}\n{data['function']}"
        elif 'module' in data:
            labels[node] = f"{node}\n{data['module']}"
        else:
            labels[node] = str(node)

    if extra_edges:
        edge_colors = ['red' if (u, v) in extra_edges else 'gray' for (u, v) in G.edges()]
    else:
        edge_colors = ['gray' for (u, v) in G.edges()]

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG for topological layout")

    # Compute topological depth
    depth = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depth[node] = 0 if not preds else 1 + max(depth[p] for p in preds)

    # Group nodes by depth
    layers = defaultdict(list)
    for node, d in depth.items():
        layers[d].append(node)

    # Position nodes by layer
    pos = {}
    layer_spacing = 2.0
    node_spacing = 2.0
    for d, nodes in layers.items():
        for i, node in enumerate(nodes):
            x = i * node_spacing
            y = -d * layer_spacing
            pos[node] = (x, y)

    # Dynamic figure size
    max_width = max(len(nodes) for nodes in layers.values())
    fig_width = max(10, max_width * 2)
    fig_height = max(6, len(layers) * 2)
    plt.figure(figsize=(fig_width, fig_height))

    # Draw
    nx.draw(
        G, pos, labels=labels, with_labels=True,
        node_color='lightblue', edge_color=edge_colors,
        node_size=600, font_size=8
    )

    # Save the figure as SVG
    plt.savefig(filename, format='svg')
    plt.close()

def parse_output(f, hw):
    """
    Parses the output file from the optimizer in the inverse pass, mapping variable names to
    technology parameters and updating them accordingly.

    Args:
        f (file-like): Opened file object containing the output to parse.

    Returns:
        None
    """
    lines = f.readlines()
    mapping = {}
    max_ind = 0
    i = 0
    while lines[i][0] != "x":
        i += 1
    while lines[i][0] == "x":
        mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
            hw.circuit_model.tech_model.base_params.symbol_table[lines[i].split(" ")[-1][:-1]]
        )
        max_ind = int(lines[i][lines[i].find("[") + 1 : lines[i].find("]")])
        i += 1
    while i < len(lines) and lines[i].find("x") != 4:
        i += 1
    i += 2
    #print(f"mapping: {mapping}, max_ind: {max_ind}")
    for _ in range(max_ind):
        key = lines[i].split(":")[0].lstrip().rstrip()
        value = float(lines[i].split(":")[2][1:-1])
        if key in mapping:
            #print(f"key: {key}; mapping: {mapping[key]}; value: {value}")
            hw.circuit_model.tech_model.base_params.tech_values[mapping[key]] = (
                value
            )
        i += 1


def write_wirelengths(wirelength_dict, path):
    """
    Serialize a dict with tuple keys (src, dst) and pandas.Series values
    to a JSON file.

    Args:
        wirelength_dict (dict[tuple[str, str], pd.Series]): wirelength data
        path (str | Path): output JSON path
    """
    json_ready = [
        {
            "src": k[0],
            "dst": k[1],
            "wirelengths": v.to_dict() if isinstance(v, pd.Series) else dict(v),
        }
        for k, v in wirelength_dict.items()
    ]

    path = Path(path)
    with path.open("w") as f:
        json.dump(json_ready, f, indent=4)
    print(f"[write_wirelengths] Wrote {len(json_ready)} wirelength entries → {path}")


def read_wirelengths(path):
    """
    Load a JSON file produced by write_wirelengths() and reconstruct
    the dict with tuple keys and pandas.Series values.

    Args:
        path (str | Path): input JSON path

    Returns:
        dict[tuple[str, str], pd.Series]
    """
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    # Handle both list-of-dicts and dict forms gracefully
    if isinstance(data, list):
        wirelengths = {
            (entry["src"], entry["dst"]): pd.Series(entry["wirelengths"])
            for entry in data
        }
    elif isinstance(data, dict):
        # If the JSON was written as {"src->dst": {...}}
        wirelengths = {}
        for k, v in data.items():
            if "->" in k:
                src, dst = k.split("->", 1)
            else:
                src, dst = k, ""
            wirelengths[(src, dst)] = pd.Series(v)
    else:
        raise ValueError("Unrecognized JSON format for wirelengths")

    print(f"[read_wirelengths] Loaded {len(wirelengths)} wirelength entries from {path}")
    return wirelengths

def netlist_plot(G, filename):
    # Create custom labels using the 'function' attribute if available
    labels = {}
    for node, data in G.nodes(data=True):
        if 'function' in data:
            labels[node] = f"{node}\n{data['function']}"
        elif 'module' in data:
            labels[node] = f"{node}\n{data['module']}"
        else:
            labels[node] = str(node)

    # dynamically adjust figure size
    max_width = max(len(nodes) for nodes in labels.values())
    fig_width = max(10, max_width * 2)
    fig_height = max(6, len(labels) * 2)
    plt.figure(figsize=(fig_width, fig_height))
    nx.draw(G, labels=labels, with_labels=True, node_color='lightblue')
    plt.savefig(filename, format='svg')
    plt.close()


if __name__ == "__main__":
    filename = "src/tmp/benchmark/parse_results/gemm_full_netlist_unfiltered.gml"
    G = nx.read_gml(filename)
    netlist_plot(G, "src/tmp/benchmark/parse_results/gemm_full_netlist_unfiltered.svg")