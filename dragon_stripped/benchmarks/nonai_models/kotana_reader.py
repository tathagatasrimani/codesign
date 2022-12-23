import os
import graphviz as gv
import numpy as np
total_cycles = 0
reads = 0
writes = 0
stage =  {"F","Rn" "Wat", "Sr", "Sw", "Wb", "Cm"}
all_reads = []
all_writes = []


def kotana_reader():
    file = open("../../vis-oos-16simd-32op-192-192reg.c0.txt",'r')
    lines = []
    global reads, writes
    for line in file.readlines():
        count = 0
        global total_cycles
        if count >= 40088 and count <= 3234212:
            instruction = line.split("\t")[-1]
            if('C\t' in line):
                # print(line,line.split("\t")[-1])
                total_cycles += int(line.split("\t")[-1])
            if len(instruction.split("="))>1:
                lines.append(instruction.split("=")[-1].split("(")[0].strip())
            if "Rn" in line:
                all_reads.append(int(line.split("\t")[-1]))
                reads+=1
            if "Wb" in line:
                all_writes.append(int(line.split("\t")[-1]))
                writes+=1
        count += 1
    return lines
# L   8970    0   10e4c r7 = vsetvli(r7)
# and ends on
# L   66345   0   10f24 bne(r28)
# {"I", "L", "S", "W","C", "E","R", "D"}
riscv_instruction_set = ['fmv.d.x','fmadd.d','vle64','vfmul.vv','vfmacc.vv','VFALU','VFXLD','vlxei64' 'addi','fld','addi','bne','add','slli','ld', "iLD", "iALU", "fLD", "iSFT", "fMUL"]
inst_energy = ['0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1','0.1']
stage_energy = []
#total_cycles = 5066

pattern1 = ['ld', 'fld', 'addi', 'slli', 'add', 'fld', 'fmadd.d','add']
pattern2 = ['fLD', 'iALU', 'iSFT', 'iALU', 'fLD', 'fMUL', 'iALU', 'iLD']
pattern3 = ['vfmacc.vv','VFALU','VFXLD','vlxei64']
pattern4 = ['addi', 'addi', 'iALU', 'iALU']

# Parse C num_cycles to increment cycle count
def kotana_graph(lines):
    energy = 0
    for inst in lines:
        if inst in riscv_instruction_set:
            index = riscv_instruction_set.index(inst)
            energy += float(inst_energy[index])
        # if inst in stage:
        #     index = riscv_instruction_set.index(inst)
        #     energy += stage_energy[index]
    # dummy energy numbers
    print("Total energy Consumption:", energy)
    print("Total Power consumption:", energy/total_cycles)

class Processor_Graph(object):
    def __init__(self, nodes):
        self.nodes = nodes
        self.name = 'Processor_Graph'
        
    def _build_visual(self, format='pdf', calls=True):
        graph = gv.Digraph(name='cluster'+self.name, format=format,
                           graph_attr={'label': self.name})
        id = 0
        i = 0
        global pattern1, pattern2, pattern3, pattern4
        pattern1_count = 0
        pattern2_count = 0
        pattern3_counts = np.zeros(len(pattern3))
        pattern4_count = 0
        pattern1_str = "Floating Matrix Add"
        pattern2_str = "Integer ALU and Integer Shift"
        pattern3_str = "Vector Matrix Multiplication"
        pattern4_str = "Integer Addition"
        while i < len(self.nodes):
            node = self.nodes[i]
      
            if node == pattern1[0]:
                if self.nodes[i+1] == pattern1[1]:
                    i+= len(pattern1)-1
                    pattern1_count+=1
            if node == pattern2[0]:   
                if self.nodes[i+1] == pattern2[1]:
                   i+= len(pattern2)-1 
                   pattern2_count+=1
            if node == pattern4[0]:
                if self.nodes[i+1] == pattern4[1]:
                    i+= len(pattern4)-1
                    pattern4_count+=1
            if node in pattern3:
                count = 0
                while self.nodes[i+1]==node:
                    count += 1
                    i+=1
                pattern3_counts[pattern3.index(node)] = pattern3_counts[pattern3.index(node)] +count
                # id +=1
                # if id>=2:
                    # graph.edge(str(id-2), str(id-1), label="next",_attributes={'style': 'dashed'})
            i+=1
        
        graph.node(str(0), label=pattern1_str+" x "+str(pattern1_count))
        graph.node("0_expanded", label=",".join(pattern1))
        
        graph.node(str(1), label=pattern2_str+" x "+str(pattern2_count))
        graph.node("1_expanded", label=",".join(pattern2))
        
        graph.node(str(2), label=pattern3_str)
        pattern3_full = ""
        for i, node in enumerate(pattern3):
            pattern3_full += node + " x " + str(pattern3_counts[i]) + "\n"
        graph.node("2_expanded", label=pattern3_full)
        
        graph.node(str(3), label=pattern4_str+" x "+str(pattern4_count))
        graph.node("3_expanded", label=",".join(pattern4))

        graph.edge(str(0), str(1), label="next",_attributes={'style': 'dashed'})
        graph.edge(str(1), str(2), label="next",_attributes={'style': 'dashed'})
        graph.edge(str(2), str(3), label="next",_attributes={'style': 'dashed'})
        
        graph.edge(str(0), "0_expanded", label="expansions",_attributes={'style': 'dashed'})
        graph.edge(str(1), "1_expanded", label="expansions",_attributes={'style': 'dashed'})
        graph.edge(str(2), "2_expanded", label="expansions",_attributes={'style': 'dashed'})
        graph.edge(str(3), "3_expanded", label="expansions",_attributes={'style': 'dashed'})
        return graph

    def build_visual(self, filepath, format, calls=True, show=True):
        """
        Build a visualisation of the CFG with graphviz and output it in a DOT
        file.

        Args:
            filename: The name of the output file in which the visualisation
                      must be saved.
            format: The format to use for the output file (PDF, ...).
            show: A boolean indicating whether to automatically open the output
                  file after building the visualisation.
        """
        graph = self._build_visual(format, calls)
        graph.render(filepath, view=False)
    
    
    
def pareto_curve():
    #total_cycles = 1800
    # scaleup = [2x, 4x, 8x] design points
    
    # bandwidth vs execution time
    
    old_mem_cycles = reads+writes
    bw_cycles = []
    for bw_avail in np.arange(400, 1000, 200):
        mem_cycles = 0
        for reads in all_reads:
            mem_bw_req = reads/cycle

            mem_cycles += mem_bw_req/bw_avail
        for writes in all_writes:
            mem_bw_req = reads/cycle

            mem_cycles += mem_bw_req/bw_avail
    bw_cycles.append(total_cycles-old_mem_cycles+new_mem_cycles)
        
    # no of cores vs execution time
    core_cycles = []
    #vfmul, vleix64, vfmacc.vv, vse64, add, sub, bne
    
    old_c_cyles = 1612
    new_c_cycles = 0
    cores = 16
    for cores in np.arange(16, 128, 32):
        new_c_cycles = old_c_cycles*cores//16
        core_cycles.append(total_cycles-old_c_cycles+new_c_cycles)
    
    # cache size vs execution time
    cache_cycles = []
    # vle64, ld, mv
    old_cache_cyles = 130
    for cache in [32,64]:
        new_cache_cycles = old_cache_cyles*cache//32
        cache_cycles.append(total_cycles-old_cache_cycles+new_cache_cycles)
        
    print(core_cycles, cache_cycles)
