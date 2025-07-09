import math
import os
import subprocess
import logging
from collections import defaultdict
import cvxpy as cp
import numpy as np
logger = logging.getLogger(__name__)

from . import cacti_util

def get_resource_name_for_file(resource_name):
    """
    Sanitize a resource name by replacing special characters with underscores for safe file naming.

    Args:
        resource_name (str): The original resource name string.

    Returns:
        str: Sanitized resource name with special characters replaced by underscores.
    """
    # replace weird characters with _
    return resource_name.replace("/", "_").replace("<", "_").replace(">", "_").replace(",", "_").replace(".", "_").replace(":", "_")

def generate_sample_memory(latency, area, clk_period, mem_type, resource_name, bandwidth):
    """
    Generate a sample memory TCL and Verilog file with specified latency, area, clock period, and bandwidth (number of ports).
    Duplicate the ports and signals in both files according to the bandwidth argument, giving each a unique name.
    """
    current_directory = os.getcwd()
    if current_directory.find("/codesign") != -1:
        current_directory = current_directory[:current_directory.find("/codesign")]
    num_cycles = math.ceil(latency/clk_period)
    input_delay = latency % clk_period # remainder of delay
    logger.info(f"num cycles for memory {resource_name}: {num_cycles}")
    resource_name_for_file = get_resource_name_for_file(resource_name)
    library_name = f"{mem_type}_{resource_name_for_file}"
    os.makedirs("src/tmp/benchmark/mem_gen", exist_ok=True)

    # --- TCL Generation ---
    with open(f"src/mem_gen/{mem_type}.tcl", "r") as f:
        lines = f.readlines()
        output = []
        ports_section = []
        pinmaps_section = []
        for line in lines:
            l = line.strip()
            # Regular replacements for the rest of the TCL
            items = l.split(' ')
            item_quantity = items[-1]
            new_line = line
            if (l.find("LIBRARY") != -1):
                new_line = line.replace(item_quantity, library_name)
            elif (l.find("WRITEDELAY") != -1) or (l.find("READDELAY") != -1):
                new_line = line.replace(item_quantity, str(input_delay))
            elif (l.find("READLATENCY") != -1) or (l.find("WRITELATENCY") != -1):
                new_line = line.replace(item_quantity, str(num_cycles))
            elif (l.find("AREA") != -1):
                new_line = line.replace(item_quantity, str(area))
            elif l.find("OUTPUT_DIR") != -1:
                new_line = line.replace("/nfs/rsghome/pmcewen", current_directory)
            elif l.find("FILENAME") != -1:
                new_line = line.replace(f"{items[2]}", f"{current_directory}/codesign/src/tmp/benchmark/mem_gen/{mem_type}_{resource_name_for_file}.v")
            output.append(new_line)

        # Insert PORTS section after PARAMETERS or after DEPTH if no PARAMETERS
        insert_idx = 0
        for idx, line in enumerate(output):
            if line.strip().startswith("PARAMETERS {"):
                # find the closing }
                for j in range(idx+1, len(output)):
                    if output[j].strip() == "}":
                        insert_idx = j+1
                        break
                break
            elif line.strip().startswith("DEPTH"):
                insert_idx = idx+1
        # Generate PORTS section
        ports_section.append("PORTS {\n")
        ports_section += ["  { NAME port_%d MODE Read  }\n" % i for i in range(bandwidth)]
        ports_section += ["  { NAME port_%d MODE Write }\n" % (i+bandwidth) for i in range(bandwidth)]
        ports_section.append("}\n")
        # Insert PINMAPS section after PORTS
        pinmaps_section.append("PINMAPS {\n")
        clk_ports = " ".join([f"port_{i}" for i in range(2*bandwidth)])
        pinmaps_section.append(f"  {{ PHYPIN clk   LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {{}} PORTS {{{clk_ports}}} }}\n")
        for i in range(bandwidth):
            pinmaps_section.append(f"  {{ PHYPIN re_{i}   LOGPIN READ_ENABLE  DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {{}} PORTS port_{i}          }}\n")
            pinmaps_section.append(f"  {{ PHYPIN radr_{i} LOGPIN ADDRESS      DIRECTION in  WIDTH addr_width PHASE {{}} DEFAULT {{}} PORTS port_{i}          }}\n")
            pinmaps_section.append(f"  {{ PHYPIN q_{i}    LOGPIN DATA_OUT     DIRECTION out WIDTH data_width PHASE {{}} DEFAULT {{}} PORTS port_{i}          }}\n")
        for i in range(bandwidth):
            idx = i + bandwidth
            pinmaps_section.append(f"  {{ PHYPIN we_{i}   LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {{}} PORTS port_{idx}          }}\n")
            pinmaps_section.append(f"  {{ PHYPIN wadr_{i} LOGPIN ADDRESS      DIRECTION in  WIDTH addr_width PHASE {{}} DEFAULT {{}} PORTS port_{idx}          }}\n")
            pinmaps_section.append(f"  {{ PHYPIN d_{i}    LOGPIN DATA_IN      DIRECTION in  WIDTH data_width PHASE {{}} DEFAULT {{}} PORTS port_{idx}          }}\n")
        pinmaps_section.append("}\n")
        # Insert sections
        output = output[:insert_idx] + ports_section + pinmaps_section + ["}\n"]
        with open(f"src/tmp/benchmark/mem_gen/{mem_type}_{resource_name_for_file}.tcl", "w") as f_new:
            f_new.writelines(output)

    # --- Verilog Generation ---
    with open(f"src/mem_gen/verilog/{mem_type}.v", "r") as f:
        vlines = f.readlines()
    # Find module line and parameter block
    module_line = None
    param_block = []
    param_start = None
    param_end = None
    for idx, line in enumerate(vlines):
        if line.strip().startswith("module "):
            module_line = line
        if line.strip().startswith("#("):
            param_start = idx
        if param_start is not None and line.strip().startswith(")"):
            param_end = idx
            break
    if module_line is None or param_start is None or param_end is None:
        raise RuntimeError("Verilog template missing module or parameter block.")
    param_block = vlines[param_start:param_end+1]
    # Generate port declarations
    port_decls = ["    input clk,\n"]
    for i in range(bandwidth):
        port_decls.append(f"    input [addr_width-1:0] radr_{i},\n")
        port_decls.append(f"    input [addr_width-1:0] wadr_{i},\n")
        port_decls.append(f"    input [data_width-1:0] d_{i},\n")
        port_decls.append(f"    input we_{i},\n")
        port_decls.append(f"    input re_{i},\n")
        port_decls.append(f"    output reg [data_width-1:0] q_{i},\n")
    if port_decls:
        port_decls[-1] = port_decls[-1].rstrip(',\n') + '\n'
    # Compose new module header
    header = []
    header.append(module_line)
    header.extend(param_block)
    header.extend(port_decls)
    header.append(");\n")
    # Copy reg declaration
    after_ports = []
    for idx in range(param_end+1, len(vlines)):
        if vlines[idx].strip().startswith("reg [data_width-1:0] mem "):
            after_ports.append(vlines[idx])
            break
    # Generate always block for all ports
    always_block = ["    always @(posedge clk) begin\n"]
    for i in range(bandwidth):
        always_block.append(f"        // Write port {i}\n")
        always_block.append(f"        if (we_{i}) begin\n")
        always_block.append(f"            mem[wadr_{i}] <= d_{i};\n")
        always_block.append(f"        end\n")
    for i in range(bandwidth):
        always_block.append(f"        // Read port {i}\n")
        always_block.append(f"        if (re_{i}) begin\n")
        always_block.append(f"            q_{i} <= mem[radr_{i}];\n")
        always_block.append(f"        end\n")
    always_block.append("    end\n")
    # Insert endmodule
    after_always = ["endmodule\n"]
    # Write the new Verilog file
    with open(f"src/tmp/benchmark/mem_gen/{mem_type}_{resource_name_for_file}.v", "w") as f_new:
        f_new.writelines(header)
        f_new.writelines(after_ports)
        f_new.writelines(always_block)
        f_new.writelines(after_always)

    tcl_file = f"src/tmp/benchmark/mem_gen/{mem_type}_{resource_name_for_file}.tcl"
    tcl_command = f"directive set {resource_name} -MAP_TO_MODULE {library_name}.{mem_type}\n"
    return tcl_file, tcl_command, library_name

class memory:
    """
    Represents a memory resource with attributes for off-chip/on-chip, depth, word width, and
    component type.

    Args:
        path_name (str): File path for the memory resource.
        name (str): Name of the memory.
        off_chip (bool): Whether the memory is off-chip.
        depth (int): number of elements in the memory.
        word_width (int): Width of each element.
        component (str): Component type string.
        mode (str): Memory operation mode.
    """
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
    """
    Parse a memory report file to extract memory resource information into memory objects.

    Args:
        filename (str): Path to the memory report file to parse.

    Returns:
        list: List of memory objects extracted from the report.
    """
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
            off_chip = False # only dealing with on chip buffers for now
            """off_chip = True
            if external_and_mode[1] == "false":
                off_chip = False
            else:
                assert external_and_mode[1] == "true" """
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

def gen_cacti_on_memories(memories, hw, bandwidths):
    memory_vals = {}
    existing_memories = {}
    for memory in memories:
        mem_file = "mem_cache" if memory.off_chip else "base_cache"
        cache_type = "main memory" if memory.off_chip else "cache"
        mem_info = (memory.depth, memory.word_width, bandwidths[memory.name])
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
                num_rw_ports=bandwidths[memory.name]
            )
            existing_memories[mem_info] = memory
        logger.info(f"config vals: {memory_vals[memory.name]}")
        logger.info(f"{cache_type} VALS: read/write time {memory_vals[memory.name]['Access time (ns)']} ns, read energy {memory_vals[memory.name]['Dynamic read energy (nJ)']} nJ, write energy {memory_vals[memory.name]['Dynamic write energy (nJ)']} nJ, leakage power {memory_vals[memory.name]['Standby leakage per bank(mW)']}")
        logger.info(f"{cache_type} cacti with: {memory.depth * memory.word_width} bytes, {memory.word_width} bus width")
        memory_vals[memory.name]["type"] = "Mem" if memory.off_chip else "Buf"
        memory_vals[memory.name]["bandwidth"] = bandwidths[memory.name]
    return memory_vals, existing_memories

def get_pre_assign_counts(bom_file, module_map):
    # calculate pre-assign PE counts to determine what memory ports are needed to saturate them
    element_counts = {}
    file = open(bom_file, "r")
    lines = file.readlines()
    i = 0
    while (i < len(lines)):
        if lines[i].strip().startswith("[Lib: assembly]"):
            break
        i += 1
    i += 1
    while (i < len(lines)):
        if lines[i].strip().startswith("[Lib: nangate") or lines[i].strip().startswith("TOTAL AREA"):
            break
        data = lines[i].strip().split()
        if data:
            module_type = data[0].split('(')[0]
            if module_type in module_map:
                # pre assign count is second to last column
                if data[-2] == 0:
                    logger.warning(f"{module_type} has zero count pre assign, setting to 1")
                if module_map[module_type] not in element_counts:
                    element_counts[module_map[module_type]] = 0
                # if multiple modules are mapped to the same module type, just add them up for count purposes
                element_counts[module_map[module_type]] += max(int(data[-2]), 1)
        i += 1
    logger.info(str(element_counts))
    return element_counts

def get_area_data(memories, max_bandwidth, hw):
    # assuming on chip memories
    # calculate memory areas for different memory types and port counts
    interval = 10
    area_data = {}
    unique_memory_types = set([memory.component for memory in memories])
    memory_depths = defaultdict(int)
    for memory in memories:
        memory_depths[memory.component] = max(memory_depths[memory.component], memory.depth)
    for memory_type in unique_memory_types:
        area_data[memory_type] = {}
        for i in range(1, min(memory_depths[memory_type], max_bandwidth)+1, interval):
            memory_vals = cacti_util.gen_vals(
                filename="base_cache",
                cache_size=memory_depths[memory_type] * memory.word_width,
                cache_type="cache",
                bus_width=memory.word_width,
                transistor_size=hw.cacti_tech_node,
                num_rw_ports=i,
            )
            area_data[memory_type][i] = memory_vals["Area (mm2)"]*1e6 # convert to um^2
    return area_data
    
    
def match_bandwidths(memories, pre_assign_counts, hw):
    max_bandwidth = max(list(pre_assign_counts.values()))
    bandwidths = {}
    logger.info(f"max bandwidth: {max_bandwidth}")
    mem_ports = cp.Variable(len(memories), pos=True)
    pe_counts = cp.Variable(len(pre_assign_counts), pos=True)
    constr = []
    tot_area = 0
    area_data = get_area_data(memories, max_bandwidth, hw)
    logger.info(f"area data: {str(area_data)}")
    for i, memory in enumerate(memories):
        # fit polynomial to area data
        ports = np.array([point[0] for point in area_data[memory.component].items()])
        area = np.array([point[1] for point in area_data[memory.component].items()])
        coeffs = np.polyfit(ports, area, deg=1)
        # add constraints on ports
        constr.append(mem_ports[i] <= min(max_bandwidth, memory.depth))
        constr.append(mem_ports[i] >= 1)
        # add polynomial to objective
        tot_area += coeffs[0]*mem_ports[i] + coeffs[1]
    for i, elem in enumerate(pre_assign_counts):
        # add constraints on counts
        constr.append(pe_counts[i] <= pre_assign_counts[elem])
        constr.append(pe_counts[i] >= 1)
        # add to objective
        tot_area += hw.params.circuit_values["area"][elem]*pe_counts[i]
    # add constraints to ensure ratios between different types of PEs are the same as in pre_assign_counts
    for i, elem in enumerate(pre_assign_counts):
        for j, other_elem in enumerate(pre_assign_counts):
            if i <= j:
                continue
            constr.append(pe_counts[i] / pre_assign_counts[elem] == pe_counts[j] / pre_assign_counts[other_elem])
    obj = cp.Minimize(cp.square(hw.area_constraint - tot_area))
    prob = cp.Problem(obj, constr)
    prob.solve()
    assert prob.status == "optimal"
    # convert value to integer for port count
    bandwidths = {memory.name: int(mem_ports.value[i]) for i, memory in enumerate(memories)}
    logger.info(f"match_bandwidths result: {bandwidths}")
    return bandwidths

    

def customize_catapult_memories(mem_rpt_file, benchmark_name, hw, pre_assign_counts): #takes in a memory report from initial catapult run
    if not os.path.exists("src/tmp/benchmark/ram_sync"):
        os.makedirs("src/tmp/benchmark/ram_sync")
    clk_period = (1 / hw.params.f) * 1e9 # ns
    memories = parse_memory_report(mem_rpt_file)

    bandwidths = match_bandwidths(memories, pre_assign_counts, hw) 
    memory_vals, existing_memories = gen_cacti_on_memories(memories, hw, bandwidths)
    tcl_commands = []
    libraries = []
    top_tcl_text = ""
    mem_seen = {} # mark off existing memories when we see them

    for memory in memories:
        mem_info = (memory.depth, memory.word_width, bandwidths[memory.name])
        cur_mem_vals = memory_vals[memory.name]
        """if mem_info in mem_seen:
            assert mem_info in existing_memories
            resource_name_for_file = get_resource_name_for_file(existing_memories[mem_info].name) 
            library_name = f"{existing_memories[mem_info].component}_{resource_name_for_file}"
            tcl_cmd = f"directive set {memory.path_name} -MAP_TO_MODULE {library_name}.{existing_memories[mem_info].component}\n"
        else:"""
        mem_seen[mem_info] = True
        tcl_file, tcl_cmd, library = generate_sample_memory(
            cur_mem_vals["Access time (ns)"], 
            cur_mem_vals["Area (mm2)"]*1e6, 
            clk_period,
            memory.component, memory.path_name, bandwidths[memory.name]
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