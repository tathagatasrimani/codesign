import math
import os
import subprocess
import logging
import shutil
logger = logging.getLogger(__name__)

from . import cacti_util

def get_resource_name_for_file(resource_name):
    # replace weird characters with _
    return resource_name.replace("/", "_").replace("<", "_").replace(">", "_").replace(",", "_").replace(".", "_").replace(":", "_")

def generate_sample_memory(latency, area, clk_period, mem_type, resource_name):
    # for setting correct filepath
    current_directory = os.getcwd()
    if current_directory.find("/codesign") != -1:
        current_directory = current_directory[:current_directory.find("/codesign")] # only get the base of the filepath

    num_cycles = math.ceil(latency/clk_period)
    input_delay = latency % clk_period # remainder of delay
    logger.info(f"num cycles for memory {resource_name}: {num_cycles}")
    resource_name_for_file = get_resource_name_for_file(resource_name)
    library_name = f"{mem_type}_{resource_name_for_file}"
    with open(f"src/mem_gen/{mem_type}.tcl", "r") as f:
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
            elif line.find("OUTPUT_DIR") != -1:
                new_line = lines[i].replace("/nfs/rsghome/pmcewen", current_directory) # assume we are working out of codesign module"""
            output.append(new_line)
        with open(f"src/tmp/benchmark/{mem_type}_{resource_name_for_file}.tcl", "w") as f_new:
            f_new.writelines(output)
        f_new.close()
    tcl_file = f"src/tmp/benchmark/{mem_type}_{resource_name_for_file}.tcl"
    tcl_command = f"directive set {resource_name} -MAP_TO_MODULE {library_name}.{mem_type}\n"
    return tcl_file, tcl_command, library_name
    
class memory:
    def __init__(self, path_name, name, off_chip, depth, word_width, component, mode):
        self.off_chip = off_chip # cache or off chip
        self.depth = depth
        self.word_width = word_width
        self.path_name = path_name
        self.name = name
        self.component = component
        self.mode = mode

    def __str__(self):
        return f"==MEMORY {self.name}==\noff chip? {self.off_chip}\ncomponent: {self.component}\ndepth: {self.depth}\nword width: {self.word_width}"

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
            #print(f"resource name: {resource_name}")
            component_and_size = lines[i+1].split()
            component = component_and_size[2]
            depth = int(component_and_size[-3])
            word_width = int(component_and_size[-1])
            
            #print(f"component and size: {component_and_size}, {component}, {depth} x {word_width}")
            external_and_mode = lines[i+2].split()
            off_chip = True
            if external_and_mode[1] == "false":
                off_chip = False
            else:
                assert external_and_mode[1] == "true"
            mode = external_and_mode[-1]
            
            if component in valid_components:
                mem = memory(resource_name, get_resource_name_for_file(resource_name), off_chip, depth, word_width, component, mode)
                logger.info(str(mem))
                memories.append(mem)

            #print(f"external and mode: {external_and_mode}, {off_chip}, {mode}")

            i += 1
            while (i < len(lines)): # move to next resource
                if lines[i].lstrip().rstrip().startswith("Resource Name"):
                    break
                i += 1
    return memories

def gen_cacti_on_memories(memories, hw):
    memory_vals = {}
    existing_memories = {}
    for memory in memories:
        mem_file = "mem_cache" if memory.off_chip else "base_cache"
        cache_type = "main memory" if memory.off_chip else "cache"
        mem_info = (memory.off_chip, memory.depth, memory.word_width)
        logger.info(f"mem info: {mem_info}")
        if mem_info in existing_memories:
            logger.info(f"reusing old mem created for {existing_memories[mem_info].name} instead of {memory.name}")
            memory_vals[memory.name] = memory_vals[existing_memories[mem_info].name]
        else:
            logger.info(f"cacti tech node: {hw.cacti_tech_node}")
            memory_vals[memory.name] = cacti_util.gen_vals(
                filename=mem_file, 
                cache_size=memory.depth * memory.word_width, 
                cache_type=cache_type,
                bus_width=memory.word_width,
                transistor_size=hw.cacti_tech_node,
            )
            existing_memories[mem_info] = memory
        logger.info(f"config vals: {memory_vals[memory.name]}")
        logger.info(f"{cache_type} VALS: read/write time {memory_vals[memory.name]['Access time (ns)']} ns, read energy {memory_vals[memory.name]['Dynamic read energy (nJ)']} nJ, write energy {memory_vals[memory.name]['Dynamic write energy (nJ)']} nJ, leakage power {memory_vals[memory.name]['Standby leakage per bank(mW)']}")
        logger.info(f"{cache_type} cacti with: {memory.depth * memory.word_width} bytes, {memory.word_width} bus width")
        memory_vals[memory.name]["type"] = "Mem" if memory.off_chip else "Buf"
    return memory_vals, existing_memories

def customize_catapult_memories(mem_rpt_file, benchmark_name, hw): #takes in a memory report from initial catapult run
    if not os.path.exists("src/tmp/benchmark/ram_sync"):
        os.makedirs("src/tmp/benchmark/ram_sync")
    memories = parse_memory_report(mem_rpt_file)
    memory_vals, existing_memories = gen_cacti_on_memories(memories, hw)
    tcl_commands = []
    libraries = []
    top_tcl_text = ""
    mem_seen = {} # mark off existing memories when we see them
    for memory in memories:
        mem_info = (memory.off_chip, memory.depth, memory.word_width)
        cur_mem_vals = memory_vals[memory.name]
        if mem_info in mem_seen:
            assert mem_info in existing_memories
            resource_name_for_file = get_resource_name_for_file(existing_memories[mem_info].name) 
            library_name = f"{existing_memories[mem_info].component}_{resource_name_for_file}"
            tcl_cmd = f"directive set {memory.path_name} -MAP_TO_MODULE {library_name}.{existing_memories[mem_info].component}\n"
        else:
            mem_seen[mem_info] = True
            tcl_file, tcl_cmd, library = generate_sample_memory(
                cur_mem_vals["Access time (ns)"], 
                cur_mem_vals["Area (mm2)"]*1e6, 
                5, #TODO: specify clk period
                memory.component, memory.path_name
            )
            top_tcl_text += f"source {tcl_file}\n"
            libraries.append(library)
        tcl_commands.append(tcl_cmd)
    with open("src/tmp/benchmark/mem_gen.tcl", "w") as f:
        f.write(top_tcl_text)
    logger.info(f"map to module directives: {tcl_commands}")
    cmd = ["catapult", "-shell", "-file", f"src/tmp/benchmark/mem_gen.tcl"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"stdout of memory generation tcl script: {res.stdout}")
    if res.returncode != 0:
        raise Exception(res.stderr)
    library_tcl_commands = []
    for library in libraries:
        library_tcl_commands.append(f"solution library add {library}\n")

    # add tcl commands to import libraries
    with open("src/tmp/benchmark/scripts/generic_libraries.tcl", "a") as f:
        f.writelines(library_tcl_commands)

    new_lines = []
    with open(f"src/tmp/benchmark/scripts/{benchmark_name}.tcl", "r") as f:
        lines = f.readlines()
        for line in lines:
            new_lines.append(line)
            if line.lstrip().rstrip() == "go assembly":
                for tcl_command in tcl_commands:
                    new_lines.append(tcl_command)
    # add map to module tcl commands to top level benchmark script
    with open(f"src/tmp/benchmark/scripts/{benchmark_name}.tcl", "w") as f:
        f.writelines(new_lines)

    return memory_vals

class dummy_class:
    def __init__(self):
        self.cacti_tech_node = 0.032
def main():
    logging.basicConfig(filename=f"src/tmp/memory.log", level=logging.INFO)
    customize_catapult_memories("src/tmp/benchmark/memories.rpt", "matmult", dummy_class())
    #generate_sample_memory(2.1, 400, 5, "ccs_ram_sync_1R1W")

if __name__ == "__main__":
    main()