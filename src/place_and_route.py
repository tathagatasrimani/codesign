import numpy as np 
import numpy.random as rn 
import matplotlib.pyplot as plt 
import random 
import matplotlib as mpl 
import math

from hardwareModel import HardwareModel
import networkx as nx 
from matplotlib.patches import Rectangle

# integrating with rest of code
# change to placement()
# place_and_route.py
# add documentation
# 

rng = rn.default_rng()
def placement(hw): # return final positions, wirelength of each edge, (add attributes to the netlist (modify))
    digraph = hw.netlist
    nodes = digraph.nodes
    edges = digraph.edges 
    node_to_index = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    areas = [None for _ in range(N)]
    newEdges = []
    averageSide = 0

    for node in digraph.nodes.data(True):
        name = node[0]
        function = node[1]['function']
        index = node_to_index[name]
        areas[index] = hw.area[function]
        averageSide += areas[index] ** 0.5

    for edge in edges:
        if (edge[1], edge[0]) in newEdges:
            continue

        newEdges.append(edge)
    
    areas[6] = 50365000.0 ###################
    areas[7] = 50365000.0 ##################
    averageSide /= N
    E = [(node_to_index[src], node_to_index[dst]) for src, dst in newEdges]
    positions = [(0, 0) for _ in range(N)]
    interval = (min(areas), max(areas))
    state, c, states, costs, minWireLength, optimalPos, wireLengths = annealing(positions, E, areas, interval, averageSide, maxsteps=10000)
    plotter2(optimalPos, areas, nodes, E)

    index_to_node = {index: node for node, index in node_to_index.items()}

    for pair in wireLengths.keys():
        newPair = (index_to_node[pair[0]], index_to_node[pair[1]])
        nx.set_edge_attributes(hw.netlist, {newPair: {'wirelength': wireLengths[pair]}})
    
    for i in range(len(optimalPos)):
        node = index_to_node[i]
        nx.set_node_attributes(hw.netlist, {node: {'position': optimalPos[i]}})

    return optimalPos, wireLengths

def annealing(positions,
              edges,
              areas,
              interval,
              init_scale,
              maxsteps=10):
    for i in range(len(positions)):
        state = init_positions(init_scale)
        positions[i] = state

    wireLengths = wirelengths(positions, edges)
    cost = cost_function(wireLengths, positions, areas)
    states, costs = [[] for _ in range(len(positions))], [cost]
    optimalPos = positions.copy()
    minWireLengths = float("inf")
    best = wireLengths.copy()

    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)

        for i in range(len(positions)):
            new_state = random_neighbour(positions[i], interval, areas[i] ** 0.5, fraction)
            positions[i] = new_state
            wireLengths = wirelengths(positions, edges)
            wireLengthSum = np.sum(list(wireLengths.values()))
            new_cost = cost_function(wireLengths, positions, areas)

            if acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
            if wireLengthSum < minWireLengths and overlapCost(positions, areas) == 0:
                minWireLengths = wireLengthSum 
                optimalPos = positions.copy()
                best = wireLengths.copy() 

            states[i].append(state)
            costs.append(new_cost)
    
    print('wirelength:', minWireLengths, '|', 'overlap cost:', overlapCost(optimalPos, areas), '|', 'pos:', optimalPos)
    return state, cost_function(wireLengths, positions, areas), states, costs, minWireLengths, optimalPos, best 

def wirelengths(x, e):
    repeats = set()
    output = {}

    for edge in e:
        if (edge[1], edge[0]) not in repeats:
            repeats.add(edge)
            position0 = x[edge[0]]
            position1 = x[edge[1]]
            wirelength = ((position0[0] - position1[0]) ** 2 + (position0[1] - position1[1]) ** 2) ** 0.5
            output[edge] = wirelength 

    return output 

def f(x, positions, areas):
    return sum(x.values()) + overlapCost(positions, areas)

def overlap(first, sec, area1, area2):
    s1 = (area1 ** 0.5) / 2
    s2 = (area2 ** 0.5) / 2
    deltaX = min([first[0] + s1, sec[0] + s2]) - max([first[0] - s1, sec[0] - s2])
    deltaY = min([first[1] + s1, sec[1] + s2]) - max([first[1] - s1, sec[1] - s2])

    if deltaX < 0 or deltaY < 0:
        return 0
    if deltaX == 0 and deltaY != 0:
        return abs(s1 * deltaY)
    if deltaX != 0 and deltaY == 0:
        return abs(s2 * deltaX)

    return deltaX * deltaY 

def overlapCost(pVectors, areas, const=1):
    totalCost = 0

    for i in range(len(pVectors) - 1):
        for j in range(i + 1, len(pVectors)):
            totalCost += overlap(pVectors[i], pVectors[j], areas[i], areas[j])

    return totalCost 

def clip(x, interval):
    a, b = interval
    return max(min(x, b), a)

def init_positions(scale):
    return rng.normal(loc=0.0, scale=scale, size=2)

def cost_function(x, positions, areas):
    return f(x, positions, areas)

def random_neighbour(x, interval, scale, fraction=1):
    amplitude = (max(interval) - min(interval)) * (1 - fraction) / 10
    deltaX = (-amplitude/2.) + amplitude * rng.normal(loc=0.0, scale=scale, size=None)
    deltaY = (-amplitude/2.) + amplitude * rng.normal(loc=0.0, scale=scale, size=None)
    return (clip(x[0] + deltaX, interval), clip(x[1] + deltaY, interval))

def acceptance_probability(cost, new_cost, cur_temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / cur_temperature)
        return p 
# temperature schedules in simulated annealing
# play with this hyperparamater
# do some reading on temperature schedules
def temperature(fraction):
    return max(0.01, min(1, 1 - fraction))

def plotter2(positions, areas, nodes, edges, scaling_factor=10 ** -9):

    G = nx.Graph()
    nodes = list(nodes)
    #multiplier = 1000000

    # Calculate the size of the side of the square based on the area
    sizes = [(area ** 0.5) * scaling_factor * 10000000 for area in areas] 
    sizes[-2] /= 100

    # sizes = []

    # for i in range(len(areas)):
    #     if i == 8:
    #         sizes[i] *= 100000
    #     else:
    #         sizes[i] *= 1000000 

    for i in range(len(positions)):
        G.add_node(nodes[i], pos=(positions[i][0] * scaling_factor, positions[i][1] * scaling_factor), size=sizes[i])

    for edge in edges:
        G.add_edge(nodes[edge[0]], nodes[edge[1]])

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = [plt.cm.viridis(random.uniform(0, 1)) for _ in range(len(edges))]
    
    plt.figure(figsize=(10, 10))

    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        plt.plot([x1, x2], [y1, y2], color=edge_colors.pop(), linewidth=1.5)
    
    ax = plt.gca()
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        size = sizes[i]
        ax.add_patch(Rectangle((x - size/2, y - size/2), size, size, color='skyblue'))
    
    label_pos = {node: (coords[0], coords[1]) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold')
    plt.axis('equal')
    plt.show()
    # matplotlib
    # instead of circle nodes, graph out square nodes (show relative size)
    # make all edges straight lines
    # areas units: nm^2

# def plotter2(positions, areas, nodes, edges, zoom_factor=100000000000000):
#     G = nx.Graph()
#     nodes = list(nodes)

#     # Calculate the size of the side of the square based on the area
#     sizes = [(area ** 0.5) for area in areas]  # No scaling factor, keep actual sizes

#     for i in range(len(positions)):
#         G.add_node(nodes[i], pos=(positions[i][0], positions[i][1]), size=sizes[i])

#     for edge in edges:
#         G.add_edge(nodes[edge[0]], nodes[edge[1]])

#     pos = nx.get_node_attributes(G, 'pos')
#     edge_colors = [plt.cm.viridis(random.uniform(0, 1)) for _ in range(len(edges))]
    
#     plt.figure(figsize=(10, 10))
    
#     # Draw edges first as straight lines
#     for edge in G.edges():
#         x1, y1 = pos[edge[0]]
#         x2, y2 = pos[edge[1]]
#         plt.plot([x1, x2], [y1, y2], color=edge_colors.pop(), linewidth=1.5)
    
#     # Draw square nodes with sizes relative to their areas
#     ax = plt.gca()
#     for i, node in enumerate(G.nodes()):
#         x, y = pos[node]
#         size = sizes[i]
#         ax.add_patch(Rectangle((x - size/2, y - size/2), size, size, color='skyblue'))
    
#     # Draw node labels
#     label_pos = {node: (coords[0], coords[1]) for node, coords in pos.items()}
#     nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold')
    
#     # Set plot limits to zoom in
#     x_values = [x for x, y in positions]
#     y_values = [y for x, y in positions]
    
#     x_center = sum(x_values) / len(x_values)
#     y_center = sum(y_values) / len(y_values)
    
#     x_range = (max(x_values) - min(x_values)) / zoom_factor
#     y_range = (max(y_values) - min(y_values)) / zoom_factor
    
#     plt.xlim(x_center - x_range / 2, x_center + x_range / 2)
#     plt.ylim(y_center - y_range / 2, y_center + y_range / 2)
    
#     plt.axis('equal')
#     plt.show()

# def plotter2(positions, areas, nodes, edges, zoom_factor=2, size_multiplier=50000000000):
#     G = nx.Graph()
#     nodes = list(nodes)

#     # Apply logarithmic scaling to the areas
#     log_areas = np.log1p(areas)  # Use np.log1p to avoid issues with zero areas
#     sizes = [(area ** 0.5) * size_multiplier for area in log_areas]  # Multiply to make sizes more visible

#     for i in range(len(positions)):
#         G.add_node(nodes[i], pos=(positions[i][0], positions[i][1]), size=sizes[i])

#     for edge in edges:
#         G.add_edge(nodes[edge[0]], nodes[edge[1]])

#     pos = nx.get_node_attributes(G, 'pos')
#     edge_colors = [plt.cm.viridis(random.uniform(0, 1)) for _ in range(len(edges))]
    
#     plt.figure(figsize=(10, 10))
    
#     # Draw edges first as straight lines
#     for edge in G.edges():
#         x1, y1 = pos[edge[0]]
#         x2, y2 = pos[edge[1]]
#         plt.plot([x1, x2], [y1, y2], color=edge_colors.pop(), linewidth=1.5)
    
#     # Draw square nodes with sizes relative to their logarithmically scaled areas
#     ax = plt.gca()
#     for i, node in enumerate(G.nodes()):
#         x, y = pos[node]
#         size = sizes[i]
#         ax.add_patch(Rectangle((x - size/2, y - size/2), size, size, color='skyblue'))
    
#     # Draw node labels
#     label_pos = {node: (coords[0], coords[1]) for node, coords in pos.items()}
#     nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold')
    
#     # Set plot limits to zoom in
#     x_values = [x for x, y in positions]
#     y_values = [y for x, y in positions]
    
#     x_center = sum(x_values) / len(x_values)
#     y_center = sum(y_values) / len(y_values)
    
#     x_range = (max(x_values) - min(x_values)) / zoom_factor
#     y_range = (max(y_values) - min(y_values)) / zoom_factor
    
#     plt.xlim(x_center - x_range / 2, x_center + x_range / 2)
#     plt.ylim(y_center - y_range / 2, y_center + y_range / 2)
    
#     plt.axis('equal')
#     plt.show()