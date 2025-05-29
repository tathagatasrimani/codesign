import logging
import os
import glob
import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_latest_log_dir():
    log_dirs = glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), "../logs/*-*-*_*-*-*")))
    log_dirs = sorted(
        log_dirs,
        key=lambda x: datetime.datetime.strptime(x.split("/")[-1], "%Y-%m-%d_%H-%M-%S"),
    )
    return log_dirs[-1]

def change_clk_period_in_script(filename, new_period):
    new_lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line= line
            if line.find("set clk_period") != -1:
                new_line = line.replace(line.split()[-1], str(new_period))
            new_lines.append(new_line)
    with open(filename, "w") as f:
        f.writelines(new_lines)

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

def topological_layout_plot(graph, reverse=False, extra_edges=None):
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
    plt.show()
