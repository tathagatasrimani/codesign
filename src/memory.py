import math
import os
import subprocess
import logging
logger = logging.getLogger(__name__)

from . import cacti_util

def generate_sample_memory(latency, area, clk_period, mem_type, resource_name):
    num_cycles = math.ceil(latency/clk_period)
    input_delay = latency % clk_period # remainder of delay
    print(num_cycles)
    resource_name_for_file = resource_name.replace("/", "_") # for file naming purposes
    library_name = f"{mem_type}_{resource_name_for_file}"
    with open(f"src/mem_gen/base/{mem_type}.tcl", "r") as f:
        lines = f.readlines()
        output = []
        for i in range(len(lines)):
            line = lines[i].lstrip().rstrip()
            new_line = lines[i]
            items = line.split(' ')
            item_quantity = items[-1]
            if (line.find("LIBRARY") != -1):
                new_line = lines[i].replace(item_quantity, library_name)
            elif (line.find("WRITEDELAY") != -1) or (line.find("READDELAY") != -1):
                new_line = lines[i].replace(item_quantity, str(input_delay))
            elif (line.find("READLATENCY") != -1) or (line.find("WRITELATENCY") != -1):
                new_line = lines[i].replace(item_quantity, str(num_cycles))
            elif (line.find("AREA") != -1):
                new_line = lines[i].replace(item_quantity, str(area))
            output.append(new_line)
        with open(f"src/mem_gen/tmp/{mem_type}_{resource_name_for_file}.tcl", "w") as f_new:
            f_new.writelines(output)
        f_new.close()
    tcl_file = f"src/mem_gen/tmp/{mem_type}_{resource_name_for_file}.tcl"
    tcl_command = f"directive set {resource_name} -MAP_TO_MODULE {library_name}.{mem_type}"
    return tcl_file, tcl_command
    
class memory:
    def __init__(self, name, off_chip, depth, word_width, component):
        self.off_chip = off_chip # cache or off chip
        self.depth = depth
        self.word_width = word_width
        self.name = name
        self.component = component

def parse_memory_report(filename):
    memories = []
    valid_components = ["ccs_ram_sync_1R1W", "ccs_ram_sync_dualport", "ccs_ram_sync_singleport"]
    with open(filename, "r") as f:
        lines = f.readlines()
        i = 0
        while(i < len(lines)):
            line = lines[i].lstrip().rstrip()
            i += 1
            if line.startswith("Memory Resources"):
                break
        assert i < len(lines)
        while(i < len(lines)):
            line = lines[i].lstrip().rstrip()
            assert line.startswith("Resource Name")
            resource_name = line.split()[-1]
            print(f"resource name: {resource_name}")
            component_and_size = lines[i+1].split()
            component = component_and_size[2]
            depth = int(component_and_size[-3])
            word_width = int(component_and_size[-1])
            
            print(f"component and size: {component_and_size}, {component}, {depth} x {word_width}")
            external_and_mode = lines[i+2].split()
            off_chip = True
            if external_and_mode[1] == "false":
                off_chip = False
            else:
                assert external_and_mode[1] == "true"
            mode = external_and_mode[-1]
            
            if component in valid_components:
                memories.append(memory(resource_name, off_chip, depth, word_width, component))

            print(f"external and mode: {external_and_mode}, {off_chip}, {mode}")

            i += 1
            while (i < len(lines)): # move to next resource
                if lines[i].lstrip().rstrip().startswith("Resource Name"):
                    break
                i += 1
    return memories

def gen_cacti_on_memories(memories):
    memory_vals = {}
    for memory in memories:
        mem_file = "mem_cache" if memory.off_chip else "base_cache"
        memory_vals[memory.name] = cacti_util.gen_vals(
            filename=mem_file, 
            cache_size=memory.depth * memory.word_width, 
            bus_width=memory.word_width
        )
    return memory_vals

def main():
    memories = parse_memory_report("../dnn-accelerator-hls-master/input_db_project.rpt")
    memory_vals = gen_cacti_on_memories(memories)
    tcl_commands = []
    top_tcl_text = ""
    for memory in memories:
        cur_mem_vals = memory_vals[memory.name]
        tcl_file, tcl_cmd = generate_sample_memory(cur_mem_vals["Access time (ns)"], cur_mem_vals["Area (mm2)"]*1e6, 5, memory.component, memory.name)
        top_tcl_text += f"source {tcl_file}\n"
        tcl_commands.append(tcl_cmd)
    with open("src/mem_gen/tmp/mem_gen.tcl", "w") as f:
        f.write(top_tcl_text)
    print(tcl_commands)
    cmd = ["catapult", "-shell", "-file", f"src/mem_gen/tmp/mem_gen.tcl"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout)
    else:
        print(res.stderr)
    #generate_sample_memory(2.1, 400, 5, "ccs_ram_sync_1R1W")

if __name__ == "__main__":
    main()