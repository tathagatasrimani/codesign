import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

def box_whiskers_plot_design(type_data: str, designs: list, units: str, title: str, show_flier: bool) -> None:
    '''
    generates boxplots
    '''
    
    box_plot = None
    rcl = pd.read_csv('results/result_rcl.csv')
    rcl[type_data] = np.log2(rcl[type_data])

    sns.set_theme(rc={'figure.figsize':(10, 15)}, style="ticks")

    flierprops = dict(marker='o', markerfacecolor='None', markersize=10,  markeredgecolor='black')
    box_plot = sns.boxplot(data=rcl, x="design", y=type_data, hue="method", palette=["red", "blue"], width=0.5, showfliers=show_flier, flierprops=flierprops)
    sns.despine(offset=10, trim=True)

    plt.setp(box_plot.artists, edgecolor = 'k')
    plt.setp(box_plot.lines, color='k')

    handles, labels = plt.gca().get_legend_handles_labels()

    # Remove duplicates
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    box_plot.set_title(title)
    box_plot.set_ylabel(units)

    if not os.path.exists("fig/"):
        os.makedirs("fig/")

    if show_flier:
        plt.savefig("fig/" + type_data + '.jpeg')
    else:
        plt.savefig("fig/" + type_data + '-noflier.jpeg')
    plt.close() 
