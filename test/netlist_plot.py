import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

def get_latest_log_folder(logs_dir):
    log_folders = glob.glob(os.path.join(logs_dir, "*"))
    valid_folders = []
    
    for folder in log_folders:
        if os.path.isdir(folder):
            folder_name = os.path.basename(folder)
            # Check if the folder name matches the date-time format
            if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', folder_name):
                valid_folders.append(folder)
    
    if not valid_folders:
        raise ValueError("No valid log folders found")
    
    return max(valid_folders, key=os.path.getctime)


def parse_and_plot_netlists(directory):
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
    plt.show()
    
    # Print counts
    # for file in files:
    #     print(f"\n{file}:")
    #     for function, count in file_function_counts[file].items():
    #         print(f"  {function}: {count}")

if __name__ == "__main__":
    print("hi")
    # default is the latest directory
    latest_dir = get_latest_log_folder("logs")

    parser = argparse.ArgumentParser(
        description="Optionally specify file -f"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=latest_dir,
        help="Log Directory is analyze.",
    )
    args = parser.parse_args()
    filename = args.file

    # if the user pastes the entire path:
    if filename.count('/') > 1:
        filename = '/'.join(filename.split('/')[-2:])

    # if the user simply puts the date
    if "logs/" not in filename:
        filename = "logs/" + filename


    print(f"Check file: {filename} {latest_dir}")
    parse_and_plot_netlists(filename)