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


def box_whiskers_plot(directory, designs, database, openroad_color, estimated_color, units, title, show_flier):  
    fig = plt.figure(figsize =(10, 15))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 

    # make iterations later
    plot_base = []
    position = 0
    for data in database:
        position += 0.4
        position_list = []
        position_list.append(position)
        for instance in data: 
            plot_base.append(ax.boxplot(instance, positions= position_list, patch_artist=True, showfliers=show_flier))

    for x in range(len(plot_base)):
        if x % 2 == 0:
            colored_box2 = color(plot_base[x], 'o', estimated_color, 'None')
        else:
            colored_box1 = color(plot_base[x], 'x', openroad_color, 'None')

    pa1 = mpatches.Patch(color=openroad_color, label='OpenROAD')
    pb1 = mpatches.Patch(color=estimated_color, label='Estimated')

    ax.legend(handles=[pa1, pb1], labels=['OpenROAD', 'Estimated'], loc='upper right', fontsize=16)

    ax.autoscale()

    xtick = []
    for x in range(len(designs)):
        xtick.append(" ")
        xtick.append(designs[x])
    ax.set_xticklabels(xtick) 
    plt.ylabel(units)
    plt.title(title)

    if show_flier:
        plt.savefig(directory + '.jpeg')
    else:
        plt.savefig(directory + '-noflier.jpeg')
    plt.close(fig)


    # cap1 = ax.boxplot(cap_data[0] ,positions= [0], patch_artist=True, showfliers=showfliers_bool)
    # cap2 = ax.boxplot(cap_data[1] ,positions= [0], patch_artist=True, showfliers=showfliers_bool)
    # colored_cap1 = color(cap1, 'x', '#0080ff', 'None')
    # colored_cap2 = color(cap2, 'o', '#003694', 'None')

    # length1 = ax.boxplot(length_data[0], positions=[0.4], patch_artist=True, showfliers=showfliers_bool)
    # length2 = ax.boxplot(length_data[1], positions=[0.4], patch_artist=True, showfliers=showfliers_bool)
    # colored_length1 = color(length1, 'x', '#ffa42e', 'None')
    # colored_length2 = color(length2, 'o', '#e34400', 'None')

    # pa2 = mpatches.Patch(color='#2994ff', label='OpenROAD') 
    # pb2 = mpatches.Patch(color='#003694', label='Estimated')  

    # pa3 = mpatches.Patch(color='#ffa42e', label='OpenROAD')
    # pb3 = mpatches.Patch(color='#e34400', label='Estimated')
    

    # ax.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3],
    #       labels=['', '', '', '', 'OpenROAD', 'Estimated'],
    #       ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
    #       loc='upper right', fontsize=16)



# 1. get acces to the def file 
# 2. calculate the distance between each macro
# 3. use the metal1 resistance and capaitane and do the math thing of square and such
# 4. plot 