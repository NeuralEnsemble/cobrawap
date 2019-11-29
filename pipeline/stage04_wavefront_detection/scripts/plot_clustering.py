import neo
import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    evts = [ev for ev in block.segments[0].events if ev.name== 'Wavefronts'][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(evts.times,
              evts.array_annotations['x_coords'],
              evts.array_annotations['y_coords'],
              c=evts.labels,
              cmap=plt.cm.nipy_spectral, s=2)

    ax.set_xlabel('time [{}]'.format(evts.times.units.dimensionality.string))
    ax.set_ylabel('x-pixel')
    ax.set_zlabel('y-pixel')
    ax.view_init(45, -75)

    plt.savefig(args.output)
