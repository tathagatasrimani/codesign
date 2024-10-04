import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

from src.sim_util import get_latest_log_dir

def parse_and_plot_netlists(directory):
    """
    directory should be the absolute path
    """
    all_functions = set()
    file_function_counts = {}
    
    # Find all files matching the pattern in the specified directory
    file_pattern = os.path.join(directory, "netlist_*.gml")
    files = glob.glob(file_pattern)
    print(file_pattern, files)
    for filename in files:
        function_counts = defaultdict(int)
        
        with open(filename, 'r') as file:
            content = file.read()
            
            # Extract all functions
            functions = re.findall(r'function "(\w+)"', content)
            
            # Count occurrences of each function
            for function in functions:
                function_counts[function] += 1
                all_functions.add(function)
        
        # Store counts for this file
        file_function_counts[os.path.basename(filename)] = function_counts
    
    # Prepare data for plotting
    files = sorted(file_function_counts.keys())
    functions = sorted(all_functions)
    
    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(15, 8))
    bottom = np.zeros(len(files))
    
    for function in functions:
        counts = [file_function_counts[file].get(function, 0) for file in files]
        p = ax.bar(files, counts, label=function, bottom=bottom)
        
        # Add count labels on the bars
        for i, count in enumerate(counts):
            if count > 0:
                ax.text(i, bottom[i] + count/2, str(count), 
                        ha='center', va='center')
        
        bottom += counts
    
    plt.title('Count of Each Function Type in Netlists')
    plt.xlabel('Netlist File')
    plt.ylabel('Count')
    plt.legend(title='Functions', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Show plot
    plt.savefig(os.path.join(os.path.dirname(__file__), "figs", f"netlist_nodes_count_{directory.split('/')[-1]}.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optionally specify file -f"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Log Directory is analyze.",
    )
    args = parser.parse_args()

    if args.file:
        filename = args.file
        # if the user simply puts the date
        if "logs/" not in filename:
            filename = os.path.normpath(os.path.join(os.path.dirname(__file__), "../logs", filename))
    else:
        filename = get_latest_log_dir()

    print(f"Check file: {filename}")
    parse_and_plot_netlists(filename)