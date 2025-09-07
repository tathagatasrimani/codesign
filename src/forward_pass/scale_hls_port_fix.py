import os
import re

def scale_hls_port_fix(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    ports_to_create = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("/// This is top function"):
            idx += 2
            break
        idx += 1
    while idx < len(lines):
        line = lines[idx]
        if line.startswith(") {"):
            break
        var = line.split()[1].split("[")[0]
        ports_to_create.append(var)
        idx += 1
    new_lines = []
    for line in lines:
        pattern = r"port=return"
        match = re.search(pattern, line)
        if match:
            new_lines.append(f"  #pragma HLS interface s_axilite port=return\n")
            print(f"Found return port")
            """for port in ports_to_create:
                new_lines.append(f"  #pragma HLS interface s_axilite port={port} bundle=ctrl\n")"""
        elif not line.strip() == "#pragma HLS inline":
            new_lines.append(line)
    with open(file_path, "w") as f:
        print(f"Writing to {file_path}")
        f.writelines(new_lines)

if __name__ == "__main__":
    scale_hls_port_fix(os.path.join(os.path.dirname(__file__), "../tmp_for_test/benchmark/resnet18_mod.cpp"))