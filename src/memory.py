import math
import os
import subprocess

def generate_sample_memory(latency, area, clk_period, mem_type):
    num_cycles = math.ceil(latency/clk_period)
    input_delay = latency % clk_period # remainder of delay
    print(num_cycles)
    with open(f"mem_gen/base/{mem_type}.tcl", "r") as f:
        lines = f.readlines()
        output = []
        for i in range(len(lines)):
            line = lines[i].lstrip().rstrip()
            new_line = lines[i]
            items = line.split(' ')
            item_quantity = items[-1]
            if (line.find("WRITEDELAY") != -1) or (line.find("READDELAY") != -1):
                new_line = lines[i].replace(item_quantity, str(input_delay))
            elif (line.find("READLATENCY") != -1) or (line.find("WRITELATENCY") != -1):
                new_line = lines[i].replace(item_quantity, str(num_cycles))
            elif (line.find("AREA") != -1):
                new_line = lines[i].replace(item_quantity, str(area))
            output.append(new_line)
        with open(f"mem_gen/tmp/{mem_type}.tcl", "w") as f_new:
            f_new.writelines(output)
        f_new.close()
    cmd = ["catapult", "-shell", "-file", f"mem_gen/tmp/{mem_type}.tcl"]
    print(cmd)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout)
    else:
        print(res.stderr)
    



def main():
    generate_sample_memory(2.1, 400, 5, "ccs_ram_sync_1R1W")

if __name__ == "__main__":
    main()