"""
Visualize the wave detection by means of clustering the detected trigger
in (time,x,y) space.
"""

import numpy as np
import quantities as pq
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import random
import warnings
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_float
from utils.neo_utils import time_slice

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--time_slice", nargs='?', type=none_or_float, default=None,
                 help="length of time_slice in seconds.")

def plot_clustering(events, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    N = len(np.unique(events.labels))
    if N:
        cmap = sns.husl_palette(N-1, h=.5, l=.6)
        cmap = random.sample([c for c in cmap], N-1)
        cmap = ListedColormap(['k']+cmap)

        ax.scatter(events.times,
                   events.array_annotations['x_coords'],
                   events.array_annotations['y_coords'],
                   c=[int(label) for label in events.labels],
                   cmap=cmap, s=2)
    else:
        warnings.warn('No trigger events to plot in clusters!')

    ax.set_xlabel('time [{}]'.format(events.times.dimensionality.string))
    ax.set_ylabel('x-pixel')
    ax.set_zlabel('y-pixel')
    ax.view_init(45, -75)
    return ax, cmap


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    evts = block.filter(name='wavefronts', objects="Event")[0]

    if len(evts):

        if args.time_slice is not None:
            asig = block.segments[0].analogsignals[0]
            t_stop = asig.t_start.rescale('s') + args.time_slice*pq.s
            evts = time_slice(evts, t_start=asig.t_start, t_stop=t_stop)

        ax, cmap = plot_clustering(evts)

    save_plot(args.output)
