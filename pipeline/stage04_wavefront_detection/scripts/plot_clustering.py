import neo
import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import random
from utils import load_neo, save_plot

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    args = CLI.parse_args()

    block = load_neo(args.data)

    evts = [ev for ev in block.segments[0].events if ev.name== 'Wavefronts'][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N = len(np.unique(evts.labels))
    cmap = sns.husl_palette(N, h=.5, l=.6)
    cmap = random.sample([c for c in cmap], N)
    cmap = ListedColormap(cmap)

    ax.scatter(evts.times,
               evts.array_annotations['x_coords'],
               evts.array_annotations['y_coords'],
               c=[int(c) for c in evts.labels],
               cmap=cmap, s=2)

    ax.set_xlabel('time [{}]'.format(evts.times.units.dimensionality.string))
    ax.set_ylabel('x-pixel')
    ax.set_zlabel('y-pixel')
    ax.view_init(45, -75)

    save_plot(args.output)
