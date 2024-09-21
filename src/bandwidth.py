import matplotlib.pyplot as plt
import pandas as pd
import os

from . import cacti_util

CACTI_DIR = "/path/to/cacti"  # Update this to your CACTI directory

# Example function to convert frequency (you should replace it with actual logic)
def convert_frequency(bus_freq):
    # Assume bus_freq is given in MHz, converting to Hz
    return bus_freq * 1e6

# Function to iterate over different bus widths and collect CACTI results
def iter_bandwidth(cacheSize=2048, memSize=131072):
    cache_vals = []
    mem_vals = []

    for i in range(1, 11):  # powers of 2 from 2 to 1024
        bus_width = 2**i

        cache_val = cacti_util.gen_vals(
                "bw_cache",
                cacheSize=cacheSize, 
                blockSize=64,
                cache_type="cache",
                bus_width=bus_width,
            )
        if isinstance(cache_val, int) and cache_val == -1:
            print(f"INVALID CACHE {i}")
        cache_vals.append(cache_val)

        print(f"{i} got through cache_val")
        mem_val = cacti_util.gen_vals(
                "bw_mem",
                cacheSize=memSize, 
                blockSize=64,
                cache_type="main memory",
                bus_width=bus_width,
            )
        if isinstance(mem_val, int) and mem_val == -1:
            print(f"INVALID MEM {i}")
        mem_vals.append(mem_val)
        print(f"{i} got through mem_val")
    
    return cache_vals, mem_vals

# Function to plot access time, total energy, and area with respect to bandwidth
def plot_bw(cache_vals, mem_vals):
    # Extract relevant values for plotting
    bus_widths = [2**i for i in range(1, 11)]  # Powers of 2 for bus widths

    print(f'CHECK CACHE_VALS: {cache_vals}')

    # For cache values
    cache_access_time = [val.get('Access time (ns)', -1) for val in cache_vals]
    cache_total_energy = [val.get('Dynamic read energy (nJ)', -1) + val.get('Dynamic write energy (nJ)', -1) for val in cache_vals]
    cache_area = [val.get('Area (mm2)', -1) for val in cache_vals]

    # For memory values
    mem_access_time = [val.get('Access time (ns)', -1) for val in mem_vals]
    mem_total_energy = [val.get('Dynamic read energy (nJ)', -1) + val.get('Dynamic write energy (nJ)', -1) for val in mem_vals]
    mem_area = [val.get('Area (mm2)', -1) for val in mem_vals]

    # Create subplots for access time, total energy, and area
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # Plot access time
    ax[0].plot(bus_widths, cache_access_time, label='Cache Access Time', marker='o')
    ax[0].plot(bus_widths, mem_access_time, label='Memory Access Time', marker='x')
    ax[0].set_title('Access Time vs Bandwidth')
    ax[0].set_xlabel('Bus Width (bits)')
    ax[0].set_ylabel('Access Time (ns)')
    ax[0].set_xscale('log')
    ax[0].legend()
    ax[0].grid(True)

    # Plot total energy
    ax[1].plot(bus_widths, cache_total_energy, label='Cache Total Energy', marker='o')
    ax[1].plot(bus_widths, mem_total_energy, label='Memory Total Energy', marker='x')
    ax[1].set_title('Total Energy vs Bandwidth')
    ax[1].set_xlabel('Bus Width (bits)')
    ax[1].set_ylabel('Total Energy (nJ)')
    ax[1].set_xscale('log')
    ax[1].legend()
    ax[1].grid(True)

    # Plot area
    ax[2].plot(bus_widths, cache_area, label='Cache Area', marker='o')
    ax[2].plot(bus_widths, mem_area, label='Memory Area', marker='x')
    ax[2].set_title('Area vs Bandwidth')
    ax[2].set_xlabel('Bus Width (bits)')
    ax[2].set_ylabel('Area (sq.mm)')
    ax[2].set_xscale('log')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

# Example of how to use these functions:
cache_vals, mem_vals = iter_bandwidth()
plot_bw(cache_vals, mem_vals)