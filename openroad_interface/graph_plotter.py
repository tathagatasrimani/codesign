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
        rcl = pd.read_csv('results/result_rcl.csv')

        sns.set_theme(rc={'figure.figsize':(10, 15)}, style="ticks")

        plt.yscale("log")
        flierprops = dict(marker='o', markerfacecolor='None', markersize=10,  markeredgecolor='black')
        box_plot = sns.boxplot(data=rcl, x="design", y=rc_or_l, hue="method", palette=["red", "blue"], width=0.5, showfliers=show_flier, flierprops=flierprops)
        sns.despine(offset=10, trim=True)

        plt.setp(box_plot.artists, edgecolor = 'k')
        plt.setp(box_plot.lines, color='k')

        handles, labels = plt.gca().get_legend_handles_labels()

        # Remove duplicates
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        box_plot.set_title(title[rc_or_l])
        box_plot.set_ylabel(units[rc_or_l])

        if not os.path.exists("fig/"):
            os.makedirs("fig/")

        if show_flier:
            plt.savefig("fig/" + rc_or_l + '.jpeg')
        else:
            plt.savefig("fig/" + rc_or_l + '-noflier.jpeg')
        plt.close() 
