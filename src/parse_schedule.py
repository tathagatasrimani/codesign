import networkx as nx
import re

from . import sim_util

class node:
    def __init__(self, id, name, tp, module, delay):
        self.id = id
        self.name = name
        self.type = tp
        self.module = module
        self.delay = delay

    def __str__(self):
        return f"========================\nNode id: {self.id}\nName: {self.name}\ntype: {self.type}\nmodule: {self.module}\n========================\n"

    def label(self):
        return f"{self.module}-{self.id}-{self.type}"
    
def convert_to_standard_dfg(graph: nx.DiGraph, node_objects, module_map):
    """
    takes the output of parse_gnt_to_graph and converts
    it to our standard dfg format for use in the rest of the flow.
    We discard catapult-specific information and convert each node
    to one of our standard operators with the correct delay value.

    Arguments:
    - graph: output of parse_gnt_to_graph
    - node_objects: mapping of node -> node class object
    - module_map: mapping from ccore module to standard operator name
    """
    modified_graph = nx.DiGraph()
    for node in graph:
        if node_objects[node].module in module_map:
            modified_graph.add_node(
                node,
                id=node_objects[node].id,
                function=module_map[node_objects[node].module],
                cost=node_objects[node].delay,
                start_time=0,
                end_time=0,
                allocation=""
            )
    return modified_graph


def parse_gnt_to_graph(file_path):
    """
    Parses a .gnt file and creates a directed graph.
    
    :param file_path: Path to the .gnt file.
    :return: A NetworkX directed graph.
    """
    G = nx.DiGraph()

    ignore_types = ["{C-CORE", "LOOP", "ASSIGN"]

    nodes = {}
    node_successors = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line.startswith("set a("):
                continue
            node_id = line[line.find('(')+1:line.find(')')]
            tokens = line.split()
            node_name = ""
            if (line.find(" NAME ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "NAME"):
                        node_name = tokens[i+1]
                        break
                assert node_name != "", f"{line}"
            node_type = None
            if (line.find(" TYPE ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "TYPE"):
                        node_type = tokens[i+1]
                        break
                assert node_type, f"{line}"
            node_module = None
            if (line.find(" MODULE ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "MODULE"):
                        node_module = tokens[i+1]
                        break
                assert node_module, f"{line}"
            node_delay = 0
            if (line.find(" DELAY ") != -1): # todo: investigate if CYCLES need to be added to delay
                for i in range(len(tokens)):
                    if (tokens[i] == "DELAY"):
                        node_delay = float(tokens[i+1][1:-1])
                #print(node_delay)

            nodes[node_id] = node(node_id, node_name, node_type, node_module, node_delay)
        

        for line in lines:
            line = line.strip()
            if not line.startswith("set a("):
                continue
            node_id = line[line.find('(')+1:line.find(')')]
            tokens = line.split(' ')
            successors = []
            if (line.find(" SUCCS ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "SUCCS"):
                        for j in range(i+1, len(tokens)):
                            if (tokens[j] == "CYCLES"): break
                            if tokens[j] in nodes:
                                successors.append(nodes[tokens[j]])
                        break
                
                
            node_successors[node_id] = successors
        
        for n in nodes:
            G.add_node(
                nodes[n].label(),
                name=nodes[n].name,
                id=nodes[n].id,
                tp=nodes[n].type,
                module=nodes[n].module
            )
        for n in nodes:
            for successor in node_successors[nodes[n].id]:
                G.add_edge(nodes[n].label(), successor.label())

        for n in nodes:
            if nodes[n].type in ignore_types:
                G.remove_node(nodes[n].label())

        
        sim_util.topological_layout_plot(G)
    
    return G

parse_gnt_to_graph("src/benchmarks/matmult/build/MatMult.v1/schedule.gnt")