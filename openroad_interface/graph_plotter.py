import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

import numpy as np


def color(plot, shape, edge_color, fill_color):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(plot[element], color=edge_color, linewidth=2)

    for flier in plot['fliers']:
        flier.set(marker=shape, markeredgecolor=edge_color, markersize=8)

    for patch in plot['boxes']:
        patch.set(facecolor=fill_color)       
        
    return plot


def box_whiskers_plot(name, showfliers_bool,res_data, cap_data, length_data):  
    fig = plt.figure(figsize =(10, 15))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 


    res1 = ax.boxplot(res_data[0], positions= [-0.4], patch_artist=True, showfliers=showfliers_bool)
    res2 = ax.boxplot(res_data[1], positions= [-0.4], patch_artist=True, showfliers=showfliers_bool)
    colored_res1 = color(res1, 'x', '#ff6161', 'None')
    colored_res2 = color(res2, 'o','#c70000', 'None')

    cap1 = ax.boxplot(cap_data[0] ,positions= [0], patch_artist=True, showfliers=showfliers_bool)
    cap2 = ax.boxplot(cap_data[1] ,positions= [0], patch_artist=True, showfliers=showfliers_bool)
    colored_cap1 = color(cap1, 'x', '#0080ff', 'None')
    colored_cap2 = color(cap2, 'o', '#003694', 'None')

    length1 = ax.boxplot(length_data[0], positions=[0.4], patch_artist=True, showfliers=showfliers_bool)
    length2 = ax.boxplot(length_data[1], positions=[0.4], patch_artist=True, showfliers=showfliers_bool)
    colored_length1 = color(length1, 'x', '#ffa42e', 'None')
    colored_length2 = color(length2, 'o', '#e34400', 'None')

    
    pa1 = mpatches.Patch(color='#ff6161', label='OpenROAD'), 
    pa2 = mpatches.Patch(color='#2994ff', label='OpenROAD'),     
    pa3 = mpatches.Patch(color='#ffa42e', label='OpenROAD'),
    pb1 = mpatches.Patch(color='#c70000', label='Estimated'),
    pb2 = mpatches.Patch(color='#003694', label='Estimated'),
    pb3 = mpatches.Patch(color='#e34400', label='Estimated')
    

    ax.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3],
          labels=['', '', '', '', 'OpenROAD', 'Estimated'],
          ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
          loc='upper right', fontsize=16)
    ax.autoscale()
    ax.set_xticklabels([" ", " Resistance", " ", "Capacitance", " ", "Length"])
    # Creating plot
    if showfliers_bool:
        plt.savefig(name + '.jpeg')
    else:
        plt.savefig(name + '-noflier.jpeg')
    plt.close(fig)
    return fig, ax


# 1. get acces to the def file 
# 2. calculate the distance between each macro
# 3. use the metal1 resistance and capaitane and do the math thing of square and such
# 4. plot 