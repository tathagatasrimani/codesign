import networkx as nx
import re

from . import sim_util

class node:
    def __init__(self, id, name, tp, module):
        self.id = id
        self.name = name
        self.type = tp
        self.module = module

    def __str__(self):
        return f"========================\nNode id: {self.id}\nName: {self.name}\ntype: {self.type}\nmodule: {self.module}\n========================\n"

    def label(self):
        return f"{self.module}-{self.id}-{self.type}"

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
            tokens = line.split(' ')
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
            nodes[node_id] = node(node_id, node_name, node_type, node_module)
        

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