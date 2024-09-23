import matplotlib.pyplot as plt
import pandas as pd
import os

from . import cacti_util
from . import CACTI_DIR


# Example function to convert frequency (you should replace it with actual logic)
def convert_frequency(bus_freq):
    # Assume bus_freq is given in MHz, converting to Hz
    return bus_freq * 1e6

# Function to iterate over different bus widths and collect CACTI results
def iter_bandwidth(bw_range, cache_size=131072, memSize=131072):
    cache_vals = []
    mem_vals = []

    for bus_width in bw_range:  # powers of 2 from 2 to 1024
        # bus_width = 2**i

        cache_val = cacti_util.gen_vals(
                "bw_cache",
                cache_size=cache_size, 
                block_size=64,
                cache_type="cache",
                bus_width=bus_width,
            )
        if isinstance(cache_val, int) and cache_val == -1:
            print(f"INVALID CACHE {i}")
        cache_vals.append(cache_val)

        print(f"{bus_width} got through cache_val")
        mem_val = cacti_util.gen_vals(
                "bw_mem",
                cache_size=memSize, 
                block_size=64,
                cache_type="main memory",
                bus_width=bus_width,
            )
        if isinstance(mem_val, int) and mem_val == -1:
            print(f"INVALID MEM {i}")
        mem_vals.append(mem_val)
        print(f"{bus_width} got through mem_val")

    import time
    time.sleep(1)
    
    return cache_vals, mem_vals

def plot_bw(bw_range, cache_vals, mem_vals, log_scale=True):
    # Extract relevant values for plotting
    bus_widths = bw_range #[2**i for i in range(1, 14)]  # Powers of 2 for bus widths (2, 4, 8, ..., 1024)

    # print(f'CHECK CACHE_VALS: {cache_vals}')

    # For cache values
    cache_access_time = [val.get('Access time (ns)', -1) for val in cache_vals]
    cache_total_energy = [val.get('Dynamic read energy (nJ)', -1) + val.get('Dynamic write energy (nJ)', -1) for val in cache_vals]
    cache_area = [val.get('Area (mm2)', -1) for val in cache_vals]

    # For memory values
    mem_access_time = [val.get('Access time (ns)', -1) for val in mem_vals]
    mem_total_energy = [val.get('Dynamic read energy (nJ)', -1) + val.get('Dynamic write energy (nJ)', -1) for val in mem_vals]
    mem_area = [val.get('Area (mm2)', -1) for val in mem_vals]

    # Create subplots for cache and memory on separate figures
    fig_cache, ax_cache = plt.subplots(3, 1, figsize=(10, 15))
    fig_mem, ax_mem = plt.subplots(3, 1, figsize=(10, 15))

    # Custom x-ticks to show powers of 2
    x_ticks = bus_widths

    # Set x-axis scale based on the log_scale parameter
    xscale = 'log' if log_scale else 'linear'

    # Plot cache data (Access Time, Total Energy, Area)
    ax_cache[0].plot(bus_widths, cache_access_time, label='Cache Access Time', marker='o')
    ax_cache[0].set_title('Cache Access Time vs Bandwidth')
    ax_cache[0].set_xlabel('Bus Width (bits)')
    ax_cache[0].set_ylabel('Access Time (ns)')
    ax_cache[0].set_xscale(xscale)
    ax_cache[0].set_xticks(x_ticks)
    if not log_scale:
        ax_cache[0].set_xticklabels(x_ticks)
    ax_cache[0].legend()
    ax_cache[0].grid(True)

    ax_cache[1].plot(bus_widths, cache_total_energy, label='Cache Total Energy', marker='o')
    ax_cache[1].set_title('Cache Total Energy vs Bandwidth')
    ax_cache[1].set_xlabel('Bus Width (bits)')
    ax_cache[1].set_ylabel('Total Energy (nJ)')
    ax_cache[1].set_xscale(xscale)
    ax_cache[1].set_xticks(x_ticks)
    if not log_scale:
        ax_cache[1].set_xticklabels(x_ticks)
    ax_cache[1].legend()
    ax_cache[1].grid(True)

    ax_cache[2].plot(bus_widths, cache_area, label='Cache Area', marker='o')
    ax_cache[2].set_title('Cache Area vs Bandwidth')
    ax_cache[2].set_xlabel('Bus Width (bits)')
    ax_cache[2].set_ylabel('Area (sq.mm)')
    ax_cache[2].set_xscale(xscale)
    ax_cache[2].set_xticks(x_ticks)
    if not log_scale:
        ax_cache[2].set_xticklabels(x_ticks)
    ax_cache[2].legend()
    ax_cache[2].grid(True)

    # Plot memory data (Access Time, Total Energy, Area)
    ax_mem[0].plot(bus_widths, mem_access_time, label='Memory Access Time', marker='x')
    ax_mem[0].set_title('Memory Access Time vs Bandwidth')
    ax_mem[0].set_xlabel('Bus Width (bits)')
    ax_mem[0].set_ylabel('Access Time (ns)')
    ax_mem[0].set_xscale(xscale)
    ax_mem[0].set_xticks(x_ticks)
    if not log_scale:
        ax_mem[0].set_xticklabels(x_ticks)
    ax_mem[0].legend()
    ax_mem[0].grid(True)

    ax_mem[1].plot(bus_widths, mem_total_energy, label='Memory Total Energy', marker='x')
    ax_mem[1].set_title('Memory Total Energy vs Bandwidth')
    ax_mem[1].set_xlabel('Bus Width (bits)')
    ax_mem[1].set_ylabel('Total Energy (nJ)')
    ax_mem[1].set_xscale(xscale)
    ax_mem[1].set_xticks(x_ticks)
    if not log_scale:
        ax_mem[1].set_xticklabels(x_ticks)
    ax_mem[1].legend()
    ax_mem[1].grid(True)

    ax_mem[2].plot(bus_widths, mem_area, label='Memory Area', marker='x')
    ax_mem[2].set_title('Memory Area vs Bandwidth')
    ax_mem[2].set_xlabel('Bus Width (bits)')
    ax_mem[2].set_ylabel('Area (sq.mm)')
    ax_mem[2].set_xscale(xscale)
    ax_mem[2].set_xticks(x_ticks)
    if not log_scale:
        ax_mem[2].set_xticklabels(x_ticks)
    ax_mem[2].legend()
    ax_mem[2].grid(True)

    # Adjust layout for cache and memory figures
    fig_cache.tight_layout()
    fig_mem.tight_layout()

    # Show plots
    plt.show()

if __name__ == "__main__":
    # Example of how to use these functions:
    bw_range = [16 * n for n in range(1, 5)]  # [2**i for i in range(1, 12)]
    cache_vals, mem_vals = iter_bandwidth(bw_range, cache_size= 2048)
    plot_bw(bw_range, cache_vals, mem_vals)
