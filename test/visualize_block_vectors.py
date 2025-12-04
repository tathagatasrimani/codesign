import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

def parse_block_graph(data, top_block_name):
    block_graph = nx.DiGraph()
    block_queue = [top_block_name]
    loop_data = {}
    while block_queue:
        block_name = block_queue.pop(0)
        #print(f"Processing block: {block_name}")
        block_data = data[block_name] if block_name in data else loop_data[block_name]
        top_object = block_data["top"] if "top" in block_data else loop_data[block_name]["loop_II"]
        for child_block_name in block_data:
            if child_block_name == "top" or child_block_name == "loop_II":
                block_graph.add_node(block_name, activity_factor=top_object["computation_activity_factor"])
            elif child_block_name.startswith("loop_1x") and child_block_name.find("rsc_delay_only") == -1:
                block_queue.append(child_block_name)
                loop_data[child_block_name] = block_data[child_block_name]
        if "calls" in top_object:
            top_object = block_data["top"] if "top" in block_data else loop_data[block_name]["loop_II"]
            for call in top_object["calls"]:
                block_queue.append(call)

    block_queue = [top_block_name]

    while block_queue:
        block_name = block_queue.pop(0)
        #print(f"Processing block: {block_name}")
        block_data = data[block_name] if block_name in data else loop_data[block_name]
        top_object = block_data["top"] if "top" in block_data else loop_data[block_name]["loop_II"]
        for child_block_name in block_data:
            if child_block_name.startswith("loop_1x") and child_block_name.find("rsc_delay_only") == -1:
                block_queue.append(child_block_name)
                block_graph.add_edge(block_name, child_block_name, label="loop")
        if "calls" in top_object:
            top_object = block_data["top"] if "top" in block_data else loop_data[block_name]["loop_II"]
            for call in top_object["calls"]:
                block_queue.append(call)
                block_graph.add_edge(block_name, call, label="function call")
    nx.write_gml(block_graph, "block_graph.gml")
        
    return block_graph

def display_block_graph(block_graph, data, top_block_name, save_dir):
    """
    Display a hierarchical graph with nodes as circles colored by activity_factor.
    The top_block_name node is highlighted with a larger size and red border.
    """
    # Try to use hierarchical layout (dot), fallback to spring layout
    pos = nx.nx_agraph.graphviz_layout(block_graph, prog='dot')
    
    # Add more horizontal spacing between nodes
    if pos:
        x_coords = [pos[node][0] for node in pos]
        x_center = sum(x_coords) / len(x_coords) if x_coords else 0
        # Scale x-coordinates to add horizontal spacing (1.5 = 50% more space)
        horizontal_spacing_factor = 0.6
        for node in pos:
            x, y = pos[node]
            # Scale relative to center to maintain layout structure
            pos[node] = (x_center + (x - x_center) * horizontal_spacing_factor, y)
    
    # Extract activity factors for coloring
    activity_factors = [block_graph.nodes[node].get('activity_factor', 1.0) 
                       for node in block_graph.nodes()]
    
    # Normalize activity factors for colormap using log scale
    min_af = 1e-1
    max_af = max(1e3, max(activity_factors))
    
    # Apply log transformation (add small epsilon to avoid log(0))
    epsilon = 1e-10
    log_activity_factors = [np.log2(af + epsilon) for af in activity_factors]
    log_min = np.log2(min_af + epsilon)
    log_max = np.log2(max_af + epsilon)
    
    # Normalize log values to 0-1 range
    normalized_af = [(log_af - log_min) / (log_max - log_min) if log_max != log_min else 0.5 
                     for log_af in log_activity_factors]
    
    # Create colormap (light red to dark red)
    cmap = plt.cm.Reds  # Light red to dark red gradient
    
    # Calculate dynamic circle/ellipse dimensions based on layout extent
    # We'll draw ellipses that appear as circles by accounting for coordinate aspect ratio
    if pos:
        x_coords = [pos[node][0] for node in pos]
        y_coords = [pos[node][1] for node in pos]
        x_range = max(max(x_coords) - min(x_coords), 5e-3) if x_coords else 1
        y_range = max(max(y_coords) - min(y_coords), 5e-3) if y_coords else 1
        
        # Calculate aspect ratio of data coordinates
        data_aspect_ratio = x_range / y_range
        
        # Base radius as a fraction of the smaller dimension
        min_range = min(x_range, y_range)
        base_radius = min_range * 0.02  # 2% of the smaller dimension
        
        # Calculate ellipse radii that will appear as circles
        # When matplotlib scales the axes, if x_range > y_range, each x-unit takes up
        # less display space. So we need larger x-radius in data coordinates.
        # If data is wider (aspect > 1), make ellipse wider in x-direction
        # If data is taller (aspect < 1), make ellipse taller in y-direction
        if data_aspect_ratio > 1:
            # Data is wider: ellipse needs larger x-radius to compensate
            ellipse_x_radius = base_radius * data_aspect_ratio
            ellipse_y_radius = base_radius
        else:
            # Data is taller: ellipse needs larger y-radius to compensate
            ellipse_x_radius = base_radius
            ellipse_y_radius = base_radius / data_aspect_ratio
    else:
        ellipse_x_radius = 30  # Fallback default
        ellipse_y_radius = 30
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges
    edge_colors = []
    edge_styles = []
    for edge in block_graph.edges():
        edge_label = block_graph.edges[edge].get('label', '')
        if edge_label == 'loop':
            edge_colors.append('blue')
            edge_styles.append('solid')
        elif edge_label == 'function call':
            edge_colors.append('green')
            edge_styles.append('dashed')
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')
    
    nx.draw_networkx_edges(block_graph, pos, ax=ax, 
                          edge_color=edge_colors, 
                          style=edge_styles,
                          alpha=0.6, arrows=True, arrowsize=20, 
                          arrowstyle='->', width=1.5)
    
    # Draw nodes with colors based on activity_factor
    node_colors = [cmap(af) for af in normalized_af]
    node_list = list(block_graph.nodes())

    # Draw all nodes as ellipses that appear as circles (accounting for coordinate aspect ratio)
    for i, node in enumerate(node_list):
        x, y = pos[node]
        #print(f"Node: {node}, x: {x}, y: {y}")
        # Use ellipse with different x and y radii to compensate for coordinate aspect ratio
        ellipse = mpatches.Ellipse((x, y), 
                                   width=2*ellipse_x_radius, 
                                   height=2*ellipse_y_radius,
                                   facecolor=node_colors[i],
                                   edgecolor='black', linewidth=2,
                                   zorder=3)
        ax.add_patch(ellipse)
    
    # Create smart labels based on node name patterns
    def create_smart_label(node_name):
        """Create a shorter, more readable label from node name."""
        # Simple names - keep as-is
        if len(node_name) <= 20:
            return node_name
        
        # Pipeline names - extract loop numbers
        if '_Pipeline_' in node_name:
            parts = node_name.split('_Pipeline_')
            base = parts[0]  # e.g., "forward_node4"
            # Extract loop numbers (e.g., "LOOP_453_1_VITIS_LOOP_455_3" -> "LOOP_453_455")
            loop_part = parts[1] if len(parts) > 1 else ""
            loop_nums = []
            for segment in loop_part.split('_'):
                if segment.isdigit() and len(segment) >= 3:  # Loop numbers are typically 3+ digits
                    loop_nums.append(segment)
            if loop_nums:
                return f"{base}\nLOOP_{'_'.join(loop_nums[:3])}"  # Show up to 3 loop numbers
            return base
        
        # Loop names - extract loop numbers
        if node_name.startswith('loop_1x_VITIS_LOOP_'):
            loop_part = node_name.replace('loop_1x_VITIS_LOOP_', '')
            loop_nums = []
            for segment in loop_part.split('_'):
                if segment.isdigit() and len(segment) >= 2:
                    loop_nums.append(segment)
            if loop_nums:
                return f"LOOP_{'_'.join(loop_nums[:3])}"
            return "loop_1x"
        
        # Fallback: truncate with ellipsis
        return node_name[:17] + '...'
    
    labels = {node: create_smart_label(node) for node in block_graph.nodes()}
    
    # Position labels outside circles (offset above, proportional to ellipse size)
    label_pos = {}
    # Use average radius for label offset
    avg_radius = (ellipse_x_radius + ellipse_y_radius) / 2
    label_offset = avg_radius * 1.5  # Offset by 1.5x the average radius
    for node in block_graph.nodes():
        x, y = pos[node]
        # Offset label above the circle
        label_pos[node] = (x, y)
    
    nx.draw_networkx_labels(block_graph, label_pos, labels, ax=ax, 
                           font_size=6, font_weight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='none', alpha=0.3))
    
    # Add edge labels
    edge_labels = {(u, v): block_graph.edges[(u, v)].get('label', '') 
                   for u, v in block_graph.edges()}
    nx.draw_networkx_edge_labels(block_graph, pos, edge_labels, ax=ax, 
                                font_size=7, alpha=0.7)
    
    # Create colorbar with log scale
    # Use small epsilon for min to avoid log(0) issues
    epsilon = 1e-10
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                              norm=LogNorm(vmin=max(min_af, epsilon), vmax=max_af))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Activity Factor (log scale)', rotation=270, labelpad=20, fontsize=12)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linestyle='solid', linewidth=1.5, 
                  label='Loop Edge'),
        plt.Line2D([0], [0], color='green', linestyle='dashed', linewidth=1.5, 
                  label='Function Call Edge')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('Block Hierarchy Graph', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    #print(f"Visualization saved to {save_dir}")
    #plt.show()

def visualize_block_vectors(filepath, top_block_name, save_dir):
    with open(filepath, "r") as f:
        data = json.load(f)
    block_graph = parse_block_graph(data, top_block_name)
    display_block_graph(block_graph, data, top_block_name, save_dir)

def visualize_all_block_vectors(block_vectors_dir_path, top_block_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(block_vectors_dir_path):
        if file.endswith(".json"):
            filename = file.split("/")[-1].split(".")[0] + ".png"
            visualize_block_vectors(os.path.join(block_vectors_dir_path, file), top_block_name, os.path.join(save_dir, filename))

def get_latest_log_dir(log_dir):
    #print(max(os.listdir(log_dir)))
    return log_dir + "/" + max(os.listdir(log_dir))

if __name__ == "__main__":
    log_dir = get_latest_log_dir("/scratch/patrick/codesign/logs")
    filepath = os.path.join(os.path.dirname(__file__), log_dir, "block_vectors")
    visualize_all_block_vectors(filepath, "gemm", os.path.join(os.path.dirname(__file__), log_dir, "block_vectors_visualization"))