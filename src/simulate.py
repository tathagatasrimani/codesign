import sys
import os
from collections import deque
import math
import json
from pathlib import Path
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv


from memory import Memory
import schedule
import dfg_algo
import hardwareModel
from hardwareModel import HardwareModel

MEMORY_SIZE = 1000000
state_graph_counter = 0

class HardwareSimulator():

    def __init__(self):
        self.memory_module = Memory(MEMORY_SIZE)
        self.data = {}
        self.cycles = 0 # counter for number of cycles
        self.main_cfg = None
        self.id_to_node = {}
        self.path = os.getcwd()
        self.data_path = []
        self.power_use = []
        self.node_intervals = []
        self.node_avg_power = {}
        self.unroll_at = {}
        self.vars_allocated = {}
        self.where_to_free = {}
        self.compute_element_to_node_id = {}
        self.compute_element_neighbors = {}
        self.memory_needed = 0
        self.nvm_memory_needed = 0
        self.cur_memory_size = 0
        self.new_graph = None
        self.mem_layers = 0
        self.transistor_size = 0
        self.pitch = 0
        self.cache_size = 0
        self.reads = 0
        self.nvm_reads = 0
        self.writes = 0
        self.total_read_size = 0
        self.total_nvm_read_size = 0
        self.total_write_size = 0
        self.max_regs_inuse = 0
        self.max_mem_inuse = 0

    def find_nearest_mem_to_scale(self, num):
        if num < 512: return 512
        if num > 536870912: return 536870912
        return 2 ** math.ceil(math.log(num, 2))

    def make_node(self, graph, id, name, ctx, opname):
        annotation = ""
        if ctx == ast.Load or ctx == ast.Store:
            annotation = "Register"
        dfg_node = dfg_algo.Node(name, opname, id, memory_links=set())
        graph.gv_graph.node(id, name + '\n' + annotation)
        graph.roots.add(dfg_node)
        graph.id_to_Node[id] = dfg_node

    def make_edge(self, graph, source_id, target_id, annotation=""):
        source, target = graph.id_to_Node[source_id], graph.id_to_Node[target_id]
        graph.gv_graph.edge(source_id, target_id, label=annotation)
        target_node = graph.id_to_Node[target_id]
        if target_node in graph.roots: graph.roots.remove(target_node)
        source.children.append(target)
        target.parents.append(source)

    def get_hw_need(self, state, hw_spec):
        '''
        state - list of operations
        hw_spec - existing hw allocation and hw specs
        '''
        hw_need = HardwareModel(id=0,bandwidth=0,mem_layers=self.mem_layers, pitch=self.pitch, transistor_size=self.transistor_size, cache_size= self.cache_size)
        mem_in_use = 0
        for op in state:
            # print(f"in get_hw_need, op: {op}")
            if not op.operation: continue
            
            # this stuff is handling some graph stuff. 
            if op.operation != "Regs":
                #print(hw_spec[op.operation], op.operation, hw_need.hw_allocated[op.operation])
                compute_element_id = hw_need.hw_allocated[op.operation] % hw_spec.hw_allocated[op.operation]
                # print(f"compute_element_id: {compute_element_id}; compute_element_to_node_id: {self.compute_element_to_node_id}")
                if len(self.compute_element_to_node_id[op.operation]) <= compute_element_id:
                    # print(f"entered ")
                    self.init_new_compute_element(op.operation)
                # print(f"after init; op: {op.operation} compute_element_to_node_id: {self.compute_element_to_node_id}")
                compute_node_id = self.compute_element_to_node_id[op.operation][compute_element_id]
                hw_op_node = self.new_graph.id_to_Node[compute_node_id]
                op.compute_id = compute_node_id
                mem_in_use += self.process_compute_element(op, self.new_graph, hw_op_node, check_duplicate=True)
            
            hw_need.hw_allocated[op.operation] += 1
            hw_spec.compute_operation_totals[op.operation] += 1
        
        self.max_regs_inuse = min(hw_spec.hw_allocated["Regs"], max(self.max_regs_inuse, hw_need.hw_allocated["Regs"]))
        self.max_mem_inuse = max(self.max_mem_inuse, mem_in_use)
        return hw_need.hw_allocated

    def cycle_sim(self, hw_inuse, hw, max_cycles):
        node_power_sum = 0
        for i in range(max_cycles):
            self.power_use.append(0)
            
            # save current state of hardware to data array
            cur_data = ""
            for elem in hw_inuse:
                power = hw.dynamic_power[elem]
                if elem == "Regs": power *= hw.power_scale[self.find_nearest_mem_to_scale(self.memory_needed)]
                if len(hw_inuse[elem]) > 0:
                    cur_data += elem + ": "
                    count = 0
                    # active power consumption of hw in use this cycle
                    for i in hw_inuse[elem]:
                        if i > 0:
                            count += 1
                            self.power_use[self.cycles] += power

                    cur_data += str(count) + "/" + str(len(hw_inuse[elem])) + " in use. || "
            self.data[self.cycles] = cur_data

            # simulate one cycle
            for elem in hw_inuse:
                for j in range(len(hw_inuse[elem])):
                    hw_inuse[elem][j] = max(0, hw_inuse[elem][j] - 1) # decrement hw in use?
            node_power_sum += self.power_use[self.cycles]
            self.cycles += 1
        return node_power_sum

    def get_matching_bracket_count(self, name):
        if name.find('[') == -1:
            return 0
        bracket_count = 0
        name = name[name.find('[')+1:]
        bracket_depth = 1
        while len(name) > 0:
            front_ind = name.find('[')
            back_ind = name.find(']')
            if back_ind == -1: break
            if front_ind != -1 and front_ind < back_ind:
                bracket_depth += 1
                name = name[front_ind+1:]
            else:
                bracket_depth -= 1
                name = name[back_ind+1:]
                if bracket_depth == 0: bracket_count += 1
        return bracket_count

    def process_compute_element_neighbor(self, neighbor, graph, op_node, context, check_duplicate):
        name = neighbor.value
        mem_size = 0
        bracket_ind = name.find('[')
        bracket_count = self.get_matching_bracket_count(name)
        if bracket_ind != -1: name = name[:bracket_ind]
        #print(name, self.memory_module.locations)
        if name in self.memory_module.locations:
            mem_loc = self.memory_module.locations[name]
            mem_size = mem_loc.size
            #print(bracket_count, mem_loc.dims, neighbor.value)
            for i in range(bracket_count):
                mem_size /= mem_loc.dims[i]
            #print(mem_size)
            if check_duplicate:
                text = name + "\nlocation: " + str(mem_loc.location)
            else:
                text = name
            dfg_node_id = dfg_algo.set_id()
            #print(dfg_node_id)
            if check_duplicate:
                anno = "size: " + str(mem_size)
            else:
                anno = ""
            if (mem_loc.location, mem_loc.size) not in op_node.memory_links:
                self.make_node(graph, dfg_node_id, text, context, neighbor.operation)
                self.make_edge(graph, dfg_node_id, op_node.id, annotation=anno)
            op_node.memory_links.add((mem_loc.location, mem_loc.size))
            #print("node made for ", name)
        return mem_size

    def process_compute_element(self, op, graph, op_node, check_duplicate):
        '''
            Determine the amount of memory in use by this operation.

            params:
                op: ??
                graph: ??
                op_node: ?? 
                check_duplicate: boolean, ??
        '''
        parents = []
        for parent in op.parents:
            parents.append(parent.operation)
        # print(f"in process_compute_element, op: {op}")
        mem_in_use = 0
        for parent in op.parents:
            if not parent.operation: continue
            if parent.operation == "Regs":
                mem_in_use += self.process_compute_element_neighbor(parent, graph, op_node, ast.Load, check_duplicate)
            else:
                if not check_duplicate: continue
                #print(op.operation, parent.operation)
                parent_compute_id = parent.compute_id
                if parent_compute_id not in self.compute_element_neighbors[op_node.id]:
                    self.make_edge(graph, parent_compute_id, op_node.id, "")
                self.compute_element_neighbors[op_node.id].add(parent_compute_id)
                self.compute_element_neighbors[parent_compute_id].add(op_node.id)
        for child in op.children:
            if not child.operation: continue
            if child.operation == "Regs":
                mem_in_use += self.process_compute_element_neighbor(child, graph, op_node, ast.Store, check_duplicate)
        return mem_in_use

    def get_dims(self, arr):
        dims = []
        if arr[0][0] == '(': # processing tuple
            dims.append(int(arr[0][1:arr[0].find(',')]))
            if len(arr) > 2:
                for dim in arr[1:-1]:
                    dims.append(int(dim[:-1]))
            if len(arr) > 1:
                dims.append(int(arr[-1][:-1]))
        else: # processing array
            dims.append(int(arr[0][1:-1]))
            if len(arr) > 2:
                for dim in arr[1:-1]:
                    dims.append(int(dim[:-1]))
            if len(arr) > 1:
                dims.append(int(arr[-1][:-1]))
        return dims

    def process_memory_operation(self, mem_op):
        var_name = mem_op[2]
        size = int(mem_op[1])
        status = mem_op[0]
        if status == "malloc":
            dims = []
            if len(mem_op) > 3:
                dims = self.get_dims(mem_op[3:])
            self.memory_module.malloc(var_name, size, dims=dims)
        elif status == "free":
            self.memory_module.free(var_name)

    # adds all mallocs and frees to vectors, and finds the next cfg node in the data path,
    # returning the index of that node
    def find_next_data_path_index(self, i, mallocs, frees):
        pattern_seek = False
        max_iters = 1
        # print(f"i: {i}, len(self.data_path): {len(self.data_path)}, self.data_path: {self.data_path}")
        while len(self.data_path[i]) != 2:
            if len(self.data_path[i]) == 0: break
            elif len(self.data_path[i]) == 1:
                if self.data_path[i][0].startswith("pattern_seek"):
                    pattern_seek = True
                    max_iters = int(self.data_path[i][0][self.data_path[i][0].rfind('_')+1:])
            if self.data_path[i][0] == "malloc": mallocs.append(self.data_path[i])
            elif self.data_path[i][0] == "free": frees.append(self.data_path[i])
            i += 1
            if i == len(self.data_path): break
        return i, pattern_seek, max_iters

    def simulate(self, cfg, node_operation_map, hw, graphs, first):
        global state_graph_counter
        cur_node = cfg.entryblock
        if first: 
            cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
            self.main_cfg = cfg
        hw_inuse = {}
        for elem in hw.hw_allocated:
            hw_inuse[elem] = [0] * hw.hw_allocated[elem]
        i = 0
        frees = []
        mallocs = []

        i, pattern_seek, max_iters = self.find_next_data_path_index(i, mallocs, frees)
        # iterate through nodes in data dependency graph
        while i < len(self.data_path):
            next_ind, pattern_seek_next, max_iters_next = self.find_next_data_path_index(i+1, mallocs, frees)
            if i == len(self.data_path): break

            # init vars for new node in cfg data path
            node_id = self.data_path[i][0]
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])
            self.node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
            start_cycles = self.cycles # for calculating average power
            num_unroll_iterations = 0
            pattern_nodes = [node_id]
            pattern_mallocs, pattern_frees = [mallocs],  [frees]
            other_mallocs, other_frees = [], []

            if pattern_seek: 
                #print("entering pattern seek")
                pattern_seek = False
                j = next_ind
                while (not pattern_seek_next) and (j+1 < len(self.data_path)):
                    #print("hello")
                    next_node_id = self.data_path[j][0]
                    pattern_nodes.append(next_node_id)
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = self.find_next_data_path_index(j+1, other_mallocs, other_frees)
                    pattern_frees.append(other_frees)
                    pattern_mallocs.append(other_mallocs)
                    other_frees = []
                    other_mallocs = []
                if pattern_seek_next:
                    next_ind = j
                else:
                    pattern_nodes = [node_id]
                #print("found pattern")
                pattern_ind = 0
                while j+1 < len(self.data_path):
                    next_node_id = self.data_path[j][0]
                    if next_node_id != pattern_nodes[pattern_ind]: break
                    pattern_ind += 1
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = self.find_next_data_path_index(j+1, [], [])
                    if pattern_ind == len(pattern_nodes):
                        num_unroll_iterations += 1
                        pattern_ind = 0
                        next_ind = j
                        if max_iters > 1 and num_unroll_iterations+1 == max_iters: 
                            break
                if num_unroll_iterations > 0: pattern_seek = True
            elif self.unroll_at[cur_node.id]:
                # unroll the node: find the next node that is not the same as the current node
                # and count consecutive number of times this node appears in the data path
                j = next_ind
                while j+1 < len(self.data_path):
                    next_node_id = self.data_path[j][0]
                    if next_node_id != node_id: break
                    num_unroll_iterations += 1
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = self.find_next_data_path_index(j+1, [], [])
                next_ind = j

            idx = 0
            while idx < len(pattern_nodes):
                cur_node = self.id_to_node[pattern_nodes[idx]]
                mallocs = pattern_mallocs[idx]
                frees = pattern_frees[idx]
                for malloc in mallocs:
                    self.process_memory_operation(malloc)

                # try to unroll after pattern_seeking
                node_iters = num_unroll_iterations
                if self.unroll_at[cur_node.id]:
                    while idx+1 != len(pattern_nodes) and pattern_nodes[idx+1] == pattern_nodes[idx]:
                        idx += 1
                        node_iters += num_unroll_iterations
                if self.unroll_at[cur_node.id] or pattern_seek:
                    for _ in range(node_iters):
                        graph = dfg_algo.dfg_per_node(cur_node)
                        node_ops = schedule.schedule_one_node(graph, cur_node) # cur_node does not get used in this.
                        for k in range(len(node_ops)):
                            node_operation_map[cur_node][k] = node_operation_map[cur_node][k] + node_ops[k]

                # node_operation_map is dict of (states -> operations)
                # cur_node appears to be a state in the data path,
                for operations in node_operation_map[cur_node]:
                    # if unroll, take each operation in a state and create more of them
                    """
                    if self.unroll_at[cur_node.id] or pattern_seek:
                        print(iters, node_iters, len(operations))
                        new_operations = operations.copy()
                        for op in operations:
                            for j in range(node_iters):
                                new_operations.append(op)
                        operations = new_operations
                    """
                    # print(f"in simulate, operations in node_operations_map: {operations}")
                    hw_need = self.get_hw_need(operations, hw)
                    state_graph_viz = gv.Graph()
                    state_graph = dfg_algo.Graph(set(), {}, state_graph_viz)
                    op_count = 0
                    for op in operations:
                        if not op.operation or op.operation == "Regs": continue
                        op_count += 1
                        compute_id = dfg_algo.set_id()
                        self.make_node(state_graph, compute_id, hardwareModel.op2sym_map[op.operation], None, hardwareModel.op2sym_map[op.operation])
                        for parent in op.parents:
                            if parent.operation and parent.operation != "Regs":
                                parent_id = dfg_algo.set_id()
                                self.make_node(state_graph, parent_id, hardwareModel.op2sym_map[parent.operation], None, hardwareModel.op2sym_map[parent.operation])
                                self.make_edge(state_graph, parent_id, compute_id, "")
                        self.process_compute_element(op, state_graph, state_graph.id_to_Node[compute_id], check_duplicate=False)
                    # if op_count > 0:
                        # Path(simulator.path + '/benchmarks/pictures/state_graphs/').mkdir(parents=True, exist_ok=True)
                        # state_graph_viz.render(self.path + '/benchmarks/pictures/state_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:] + '_' + str(state_graph_counter), view = True)
                    state_graph_counter += 1

                    max_cycles = 0
                    # allocate hardware for operations in this state and calculate max cycles
                    for elem in hw_need:
                        latency = hw.latency[elem]
                        if elem == "Regs": latency *= hw.latency_scale[self.find_nearest_mem_to_scale(self.memory_needed)]
                        num_elem_needed = hw_need[elem]
                        if num_elem_needed == 0: continue
                        # check some flag for dynamic and set hw.hw_allocated = hw_need?
                        if hw.dynamic_allocation:
                            if hw.hw_allocated[elem] < hw_need[elem]:
                                hw_inuse[elem] = [0] * hw_need[elem]
                                hw.hw_allocated[elem] = hw_need[elem]
                        if hw.hw_allocated[elem] == 0 and num_elem_needed > 0:
                            raise Exception("hardware specification insufficient to run program")
                        cur_cycles_needed = int(math.ceil(num_elem_needed / hw.hw_allocated[elem]) * latency)
                        #print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(num_elem_needed) + ')')
                        max_cycles = max(cur_cycles_needed, max_cycles) # identify bottleneck element for this node.
                        j = 0
                        while num_elem_needed > 0:
                            hw_inuse[elem][j] += latency # this keeps getting incremented, never reset.
                            j = (j + 1) % hw.hw_allocated[elem]
                            num_elem_needed -= 1
                    self.node_avg_power[node_id] += self.cycle_sim(hw_inuse, hw, max_cycles)
                idx += 1
            
            if self.cycles - start_cycles > 0: self.node_avg_power[node_id] /= self.cycles - start_cycles
            self.node_intervals[-1][1][1] = self.cycles
            for free in frees:
                #print(free)
                if free[2] in self.memory_module.locations: 
                    self.process_memory_operation(free)
            mallocs = []
            frees = []
            i = next_ind
            pattern_seek = pattern_seek_next
            max_iters = max_iters_next

        # add all passive power at the end.
        # This is done here for the dynamic allocation case where we don't know how many 
        # compute elements we need until we run the program.
        passive_power = 0
        for elem in hw.hw_allocated:
            passive_power += hw.leakage_power[elem] * hw.hw_allocated[elem]
        
        for c in range(self.cycles):
            self.power_use[c] += passive_power

        
        print("done with simulation")
        # Path(simulator.path + '/benchmarks/pictures/memory_graphs').mkdir(parents=True, exist_ok=True)
        # self.new_graph.gv_graph.render(self.path + '/benchmarks/pictures/memory_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:], view = True)
        return self.data

    def set_data_path(self):
        with open(self.path + '/instrumented_files/output.txt', 'r') as f:
            # f_new = open(self.path + '/instrumented_files/output_free.txt', 'w+')
            src = f.read()
            l = src.split('\n')
            split_lines = [l_.split() for l_ in l] # idk what's happening here.

            last_line = '-1'
            last_node = '-1'
            valid_names = set()
            nvm_vars = {}

            # count reads and writes on first pass.
            for i in range(len(split_lines)):
                item = split_lines[i]
                if len(item) < 2: continue
                if item[0] == "malloc":
                    valid_names.add(item[2])
                if (item[-2] != "Read" and item[-2] != "Write"): continue
                var_name = item[0]
                if var_name not in valid_names and "NVM" not in var_name: continue 
                if item[-2] == "Read":
                    if "NVM" in item[0]:
                        self.nvm_reads += 1
                        self.total_nvm_read_size += int(item[-1])
                        if item[0] not in nvm_vars.keys():
                            nvm_vars[item[0]] = int(item[-1])
                        else:
                            nvm_vars[item[0]] = max(nvm_vars[item[0]], int(item[-1]))
                    else:
                        self.reads += 1
                        self.total_read_size += int(item[-1])
                        self.where_to_free[var_name] = i+1
                else: 
                    self.writes += 1
                    self.total_write_size += int(item[-1])
                    self.where_to_free[var_name] = i

            # second pass, construct trace that simulator follows.
            for i in range(len(split_lines)):
                item = split_lines[i]
                # if not (len(item) >= 3 and item[0] == "malloc" and item[2] not in self.vars_allocated):
                #     f_new.write(l[i] + '\n')
                vars_to_pop = []
                for var_name in self.where_to_free:
                    if self.where_to_free[var_name] == i:
                        var_size = self.vars_allocated[var_name]
                        # f_new.write("free " + str(var_size) + " " + var_name + "\n")
                        self.data_path.append(["free", str(var_size), var_name])
                        vars_to_pop.append(var_name)
                        self.cur_memory_size -= var_size
                        self.vars_allocated.pop(var_name)
                for var_name in vars_to_pop:
                    self.where_to_free.pop(var_name)
                if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                    last_node = item[0]
                    last_line = item[1]
                    self.data_path.append(item)      
                elif len(item) == 1 and item[0].startswith('pattern_seek'):
                    self.data_path.append(item)
                elif len(item) >= 3 and item[0] == "malloc":
                    if item[2] in self.vars_allocated:
                        if int(item[1]) == self.vars_allocated[item[2]]: 
                            continue
                        # else: 
                            # f_new.write("free " + str(self.vars_allocated[item[2]]) + " " + var_name + "\n")
                            # f_new.write(l[i] + '\n')
                        self.cur_memory_size -= int(self.vars_allocated[item[2]])
                    self.data_path.append(item)
                    self.vars_allocated[item[2]] = int(item[1])
                    #print(self.vars_allocated)
                    self.cur_memory_size += int(item[1])
                    self.memory_needed = max(self.memory_needed, self.cur_memory_size)
        # print(f"data_path: {self.data_path}")
        self.nvm_memory_needed = sum(nvm_vars.values())
        print(f"memory needed: {self.memory_needed} bytes")
        print(f"nvm memory needed: {self.nvm_memory_needed} bytes")

    def simulator_prep(self, benchmark):
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        node_operation_map = schedule.schedule(cfg, graphs)
        self.set_data_path()
        for node in cfg: self.id_to_node[str(node.id)] = node
        #print(self.id_to_node)
        return cfg, graphs, node_operation_map

    def init_new_compute_element(self, compute_unit):
        '''
        Adds to some graph shit and adds ids and stuff
        params:
            compute_unit: str with std cell name, eg 'Add', 'Eq', 'LShift', etc.
        '''
        compute_id = dfg_algo.set_id()
        self.make_node(self.new_graph, compute_id, hardwareModel.op2sym_map[compute_unit], None, hardwareModel.op2sym_map[compute_unit])
        self.compute_element_neighbors[compute_id] = set()
        self.compute_element_to_node_id[compute_unit].append(compute_id)

def main():
    print(f"Running simulator for {args.benchmark.split('/')[-1]}")
    simulator = HardwareSimulator()
    cfg, graphs, node_operation_map = simulator.simulator_prep(args.benchmark)

    hw = HardwareModel(cfg='aladdin')

    simulator.transistor_size = hw.transistor_size # in nm
    simulator.pitch = hw.pitch # in um
    simulator.mem_layers = hw.mem_layers
    if simulator.memory_needed < 1000000:
        simulator.cache_size = 1
    elif simulator.memory_needed < 2000000:
        simulator.cache_size = 2
    elif simulator.memory_needed < 4000000:
        simulator.cache_size = 4
    elif simulator.memory_needed < 8000000:
        simulator.cache_size = 8
    else: 
        simulator.cache_size = 16

   
    new_gv_graph = gv.Graph()
    simulator.new_graph = dfg_algo.Graph(set(), {}, new_gv_graph)
    for elem in hw.hw_allocated:
        if elem == "Regs": continue
        simulator.compute_element_to_node_id[elem] = []
        # looks like there's a lot of setup stuff that depends on the amount of hw allocated.
        for i in range(hw.hw_allocated[elem]):
            simulator.init_new_compute_element(elem)
            # compute_id = dfg_algo.set_id()
            # simulator.make_node(simulator.new_graph, compute_id, hardwareModel.op2sym_map[key], None, hardwareModel.op2sym_map[key])
            # simulator.compute_element_neighbors[compute_id] = set()
            # simulator.compute_element_to_node_id[key].append(compute_id)
    # print(f"before start sim: compute_elements_to_node_id: {simulator.compute_element_to_node_id}")
    
    data = simulator.simulate(cfg, node_operation_map, hw, graphs, True)

    area = 0
    for elem in hw.hw_allocated:
        area += hw.hw_allocated[elem] * hw.area[elem]
    print(f"compute area: {area * 1e-6} um^2")

    # print stats
    print("total number of cycles: ", simulator.cycles)
    print(f"Avg Power: {1e-6 * sum(simulator.power_use) / simulator.cycles} mW")
    # print(f"total energy {sum(simulator.power_use)} nJ")
    print("total volatile reads: ", simulator.reads)
    print("total volatile read size: ", simulator.total_read_size)
    print("total nvm reads: ", simulator.nvm_reads)
    print("total nvm read size: ", simulator.total_nvm_read_size)
    print("total writes: ", simulator.writes)
    print("total write size: ", simulator.total_write_size)
    print("total compute element usage: ", hw.compute_operation_totals)
    print("max regs in use: ", simulator.max_regs_inuse)
    print(f"max memory in use: {simulator.max_mem_inuse} bytes")

    # save some dump of data to json file
    names = args.benchmark.split('/')
    if not args.notrace:
        text = json.dumps(data, indent=4)
        Path(simulator.path + '/benchmarks/json_data').mkdir(parents=True, exist_ok=True)
        with open(simulator.path + '/benchmarks/json_data/' + names[-1], 'w') as fh:
            fh.write(text)
    t = []
    for i in range(len(simulator.power_use)):
        t.append(i)
    plt.plot(t,simulator.power_use)
    plt.title("power use for " + names[-1])
    plt.xlabel("Cycle")
    plt.ylabel("Power")
    Path(simulator.path + '/benchmarks/power_plots').mkdir(parents=True, exist_ok=True)
    plt.savefig("benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    plt.clf() 
    print("done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Simulate',
                    description='Runs a hardware simulation on a given benchmark and technology spec',
                    epilog='Text at the bottom of help')
    parser.add_argument('benchmark', metavar='B', type=str)
    parser.add_argument('--notrace', action='store_true')

    args = parser.parse_args()
    # print(f"args: {args.benchmark}, {args.notrace}")

    main()