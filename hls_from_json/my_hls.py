from staticfg import CFGBuilder
import json
import graphviz
import re
from collections import deque

op2sym_map = {
    "And": "and",
    "Or": "or",
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "FloorDiv": "//",
    "Mod": "%",
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitXor": "^",
    "BitAnd": "&",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "IsNot": "!=",
    "USub": "-",
    "UAdd": "+",
    "Not": "!",
    "Invert": "~",
}
delimiters = (
    "+",
    "-",
    "*",
    "//",
    "%",
    "=",
    ">>",
    "<<",
    "<",
    "<=",
    ">",
    ">=",
    "!=",
    "~",
    "!",
    "^",
    "&",
)
regexPattern = "|".join(map(re.escape, delimiters))

latency = {
    "And": 1,
    "Or": 1,
    "Add": 4,
    "Sub": 4,
    "Mult": 5,
    "FloorDiv": 16,
    "Mod": 3,
    "LShift": 0.70,
    "RShift": 0.70,
    "BitOr": 0.06,
    "BitXor": 0.06,
    "BitAnd": 0.06,
    "Eq": 1,
    "NotEq": 1,
    "Lt": 1,
    "LtE": 1,
    "Gt": 1,
    "GtE": 1,
    "USub": 0.42,
    "UAdd": 0.42,
    "IsNot": 1,
    "Not": 0.06,
    "Invert": 0.06,
    "Regs": 1,
}
energy = {}
power = {
    "And": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Or": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Add": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Sub": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Mult": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "FloorDiv": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "Mod": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "LShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "RShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "BitOr": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitXor": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitAnd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Eq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "NotEq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Lt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "LtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Gt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "GtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "USub": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "UAdd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "IsNot": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Not": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Invert": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Regs": [7.936518e-03, 1.062977e-03, 8.999495e-03, 8.999495e-03, 7.395312e-05],
}
max_hw = {}
hw_allocated = []
cycles = 0
energy = 0
flag = False # checks if the previous node was a for loop statement
num_loops = 1

def schedule(expr):
    hw_need = {}
    num_cycles = 0
    energy_need = 0
    for key in op2sym_map.keys():
        hw_need[key] = 0
    strs = re.split(regexPattern, expr)
    if strs.count("") > 0:
        strs.remove("")
    num_vars = len(strs)
    for i, op in enumerate(op2sym_map.values()):
        hw_need[list(op2sym_map.keys())[i]] += expr.count(op)
        num_cycles += hw_need[list(op2sym_map.keys())[i]]*latency[list(op2sym_map.keys())[i]]
        energy_need += hw_need[list(op2sym_map.keys())[i]]*power[list(op2sym_map.keys())[i]][2]*latency[list(op2sym_map.keys())[i]]
    hw_need["Regs"] = num_vars
    return num_cycles, hw_need, energy_need

def overlap(int1, int2):
    return max(int1[0], int2[0]) < min(int1[1], int2[1])

def buddies_of_buddies(node_visited, buddies):
    new_buddies = []
    for i in range(len(buddies)):
        new_buddies.append([])
        new_buddies[-1].append([buddies[i][0]])
        for j in range(1, len(buddies[i])):
            for k in range(len(new_buddies[-1])):
                flag = True
                for l in range(1, len(new_buddies[-1][k])):
                    if not overlap(node_visited[new_buddies[-1][k][l]], node_visited[buddies[i][j]]):
                        flag = False
                if flag: # all nodes in new_buddies[-1][k] must mutually overlap
                    new_buddies[-1][k].append(buddies[i][j])           
            new_buddies[-1].append([buddies[i][0]])  
    return new_buddies

def get_buddies(node_visited):
    buddies = []
    for node in node_visited:
        buddies.append([node])
    for i in range(len(buddies)):
        for j in range(i + 1, len(buddies)):
            if overlap(node_visited[buddies[i][0]], node_visited[buddies[j][0]]):
                buddies[i].append(buddies[j][0])
                buddies[j].append(buddies[i][0])
    return buddies

def update_hw(old_hw, new_hw):
    old_hw = {
        key: max(value, new_hw[key])
        for key, value in old_hw.items()
    }
    return old_hw

def get_max_hw(buddies, max_hw):
    for i in range(len(buddies)):
        buddy_hw = {}
        for key in op2sym_map.keys():
            buddy_hw[key] = 0
        buddy_hw["Regs"] = 0
        for j in range(len(buddies[i])):
            for hw in hw_allocated[buddies[i][j]]:
                buddy_hw[hw] += hw_allocated[buddies[i][j]][hw]
        max_hw = update_hw(max_hw, buddy_hw)
    return max_hw

def init_hw(hw):
    for key in op2sym_map.keys():
        hw[key] = 0
    hw["Regs"] = 0
    return hw

def make_reverse(node_map):
    reverse_map = {}
    for node in node_map:
        for next_node in node_map[node]:
            if reverse_map.get(next_node) == None:
                reverse_map[next_node] = [node]
            else:
                reverse_map[next_node].append(node)
    return reverse_map

def make_graph(j, asap):
    node_map = {}
    for edge in j['edges']:
        if node_map.get(edge['tail']) == None:
            node_map[edge['tail']] = []
        node_map[edge['tail']].append(edge['head'])
    if not asap: node_map = make_reverse(node_map)
    return node_map

def schedule_node_hw(node, j, update_energy):
    global hw_allocated, energy
    node_cycles = 0
    for op in j['objects'][node[0]]['_ldraw_']:
        if op['op'] != 'T': continue
        num_cycles, hw_need, energy_need = schedule(op['text'])
        hw_allocated[-1] = update_hw(hw_allocated[-1], hw_need)
        node_cycles += num_cycles
        if update_energy: energy += energy_need
    return node_cycles

def schedule_forward(j, nodes, node_map):
    global cycles, max_hw, energy, hw_allocated
    # keep track of which nodes already visited so they are not scheduled twice
    node_visited = {}
    for sub_graph in nodes:
        max_cycles = 0
        nodeQ = deque([])
        nodeQ.appendleft([sub_graph[0], 0]) #node, cycles
        while len(nodeQ) != 0:
            node = nodeQ.pop()
            hw_allocated.append(init_hw({}))
            if node_visited.get(node[0]) != None: continue
            node_visited[node[0]] = [node[1]]

            node[1] += schedule_node_hw(node, j, True)
            cycles = max(cycles, node[1])
            
            if node_map.get(node[0]) != None: 
                for i in range(len(node_map[node[0]])):
                    nodeQ.appendleft([node_map[node[0]][i], node[1]])
            node_visited[node[0]].append(node[1])
        cycles = max(cycles, max_cycles)

    return node_visited

def schedule_reverse(j, nodes, node_map):
    global cycles, max_hw, energy, hw_allocated
    # keep track of which nodes already visited so they are not scheduled twice
    node_visited = {}
    for sub_graph in nodes:
        max_cycles = 0
        nodeQ = deque([])
        for i in sub_graph:
            nodeQ.appendleft([i, 0, 0]) #node, initial cycles, resulting cycles
        while len(nodeQ) != 0:
            node = nodeQ.pop()
            hw_allocated.append(init_hw({}))
            update_energy = False # only update energy the first time you visit a node
            if node_visited.get(node[0]) == None: 
                update_energy = True
                node_visited[node[0]] = [node[1], node[2]]
            elif node[1] < node_visited[node[0]][0]:
                continue

            #keep track of the total cycles for the operations in this node
            node[2] = node[1] + schedule_node_hw(node, j, update_energy)
            max_cycles = max(max_cycles, node[2])

            if node_map.get(node[0]) != None: #if there are other nodes after this one, queue them up
                for i in range(len(node_map[node[0]])):
                    if node_map[node[0]][i] < node[0]:
                        nodeQ.appendleft([node_map[node[0]][i], node[2], node[2]])
            #update node_visited for this node only if the current start time is later than the former (already checked)
            node_visited[node[0]] = [node[1], node[2]]
        cycles = max(cycles, max_cycles)

    for node in node_visited:
        node_visited[node][0] = (node_visited[node][0] - cycles) * -1
        node_visited[node][1] = (node_visited[node][1] - cycles) * -1
        node_visited[node][0], node_visited[node][1] = node_visited[node][1], node_visited[node][0]
    return node_visited

def schedule_nodes(j, nodes, node_map, asap):
    if asap:
        return schedule_forward(j, nodes, node_map)
    else:
        return schedule_reverse(j, nodes, node_map)

def main():
    global cycles, max_hw, energy, flag, hw_allocated
    file = open('exampleCFG.json')
    j = json.load(file)
    max_hw = init_hw(max_hw)
    # make list of nodes for each subgraph
    nodes = []
    for i in range(j['_subgraph_cnt']):
        nodes.append(j['objects'][i]['nodes'])

    asap = False # either asap or alap scheduling

    # make map entry for each node to each of its connecting nodes (reverse it if alap)
    node_map = make_graph(j, asap)
    
    # schedule node intervals and update node hardware for either asap or alap objective
    node_visited = schedule_nodes(j, nodes, node_map, asap)

    # find all mutually overlapping intervals, and set max_hw accordingly
    new_buddies = buddies_of_buddies(node_visited, get_buddies(node_visited))
    for i in range(len(new_buddies)):
        possible_max = get_max_hw(new_buddies[i], init_hw({}))
        for key in possible_max:
            max_hw[key] = max(max_hw[key], possible_max[key])
    print("cycles : ", cycles)
    print("energy : ", energy)
    print("hardware: ", max_hw)

if __name__ == "__main__":
    main()