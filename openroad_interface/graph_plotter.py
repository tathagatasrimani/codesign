import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

def box_whiskers_plot(designs: list, units: list, title: list, show_flier: bool) -> None:
    '''
    generates boxplots jpegs
    param:
        designs: list of designs being evaluated
        units: specific units of the y axis
        title: title of the graph
        show_flier: turn outlier points on or off
    '''
    
    for rc_or_l in ["res", "cap", "length"]:
        box_plot = None
        rcl = pd.read_csv('openroad_interface/results/result_rcl.csv')

        sns.set_theme(rc={'figure.figsize':(4, 7)}, style="ticks")

        plt.yscale("log")
        flierprops = dict(marker='o', markerfacecolor='None', markersize=10,  markeredgecolor='black')
        box_plot = sns.boxplot(data=rcl, x="design", y=rc_or_l, hue="method", palette=["red", "blue"], width=0.5, showfliers=show_flier, flierprops=flierprops)
        sns.despine(offset=10, trim=True)

        plt.setp(box_plot.artists, edgecolor = 'k')
        plt.setp(box_plot.lines, color='k')
        box_plot.set_title(title[rc_or_l])
        box_plot.set_ylabel(units[rc_or_l])
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.tight_layout(pad=3.0)
        if not os.path.exists("openroad_interface/fig/"):
            os.makedirs("openroad_interface/fig/")

        if show_flier:
            plt.savefig("openroad_interface/fig/" + rc_or_l + '.jpeg', dpi=1200)
        else:
            plt.savefig("openroad_interface/fig/" + rc_or_l + '-noflier.jpeg', dpi=1200)
        plt.close() 


def bar_graph_cacti(data_list, data_x, name, unit, lower_lim, upper_lim, color):
    data = pd.DataFrame({
    name + " (" + unit + ")": data_list,
    'Method': data_x
    })
    plt.figure(figsize=(4, 4))
    ax = sns.barplot(x='Method', y=name + " (" + unit + ")", data=data,  color=color)
    plt.title(name + ' utilizing \n different methods')
    plt.autoscale() 
    ax.set_ylim(lower_lim, upper_lim)
    plt.tight_layout(pad=3.0)

    if not os.path.exists("openroad_interface/fig/"):
        os.makedirs("openroad_interface/fig/")

    plt.savefig("openroad_interface/fig/" + name + '.jpeg', dpi=1200)
    plt.close()