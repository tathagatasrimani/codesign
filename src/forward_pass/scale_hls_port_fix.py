import os
import re

def scale_hls_port_fix(file_path, benchmark_name, pytorch):
    with open(file_path, "r") as f:
        lines = f.readlines()
    ports_to_create = []
    idx = 0
    # TODO remove pytorch flag
    top_function_name = benchmark_name if not pytorch else "forward"
    while idx < len(lines):
        line = lines[idx]
        if line.find(f" {top_function_name}(") != -1:
            idx += 1
            break
        idx += 1
    assert idx < len(lines)
    while idx < len(lines):
        line = lines[idx]
        if line.startswith(") {"):
            break
        var = line.split()[1].split(",")[0].split("[")[0]
        ports_to_create.append(var)
        idx += 1
    new_lines = []
    for line in lines:
        pattern = r"port=return"
        match = re.search(pattern, line)
        if match:
            #new_lines.append(f"  #pragma HLS interface s_axilite port=return\n")
            new_lines.append(line)
            print(f"Found return port")
            for port in ports_to_create:
                new_lines.append(f"  #pragma HLS interface s_axilite port={port} bundle=ctrl\n")
        elif not line.strip() == "#pragma HLS inline":
            new_lines.append(line)
    with open(file_path, "w") as f:
        print(f"Writing to {file_path}")
        f.writelines(new_lines)
    
    return top_function_name

if __name__ == "__main__":
    scale_hls_port_fix(os.path.join(os.path.dirname(__file__), "../tmp_for_test/benchmark/resnet18_mod.cpp"))