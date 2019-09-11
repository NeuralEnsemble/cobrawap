import os
import sys
import argparse
import matplotlib.pyplot as plt
import neo
import numpy as np

if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--nix_file",        nargs='?', type=str)
    CLI.add_argument("--frame_folder",  nargs='?', type=str)
    CLI.add_argument("--frame_name",    nargs='?', type=str)
    CLI.add_argument("--frame_format",  nargs='?', type=str)
    CLI.add_argument("--t_start",       nargs='?', type=float)
    CLI.add_argument("--t_stop",        nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.nix_file) as io:
        up_transitions = io.read_block().segments[0].analogsignals

    frame_start = int(args.t_start * logMUA[0].sampling_rate.rescale('1/s'))
    frame_stop = int(args.t_stop * logMUA[0].sampling_rate.rescale('1/s'))
    times = logMUA[0].times[frame_start:frame_stop]

    shape = [len(times),
             logMUA[0].annotations['grid_size'][0],
             logMUA[0].annotations['grid_size'][1]]
    logMUA_array = np.zeros(shape)
    for asig in logMUA:
        x, y = asig.annotations['coordinates']
        logMUA_array[frame_start:frame_stop,x,y] = \
                            np.squeeze(asig.as_array()[frame_start:frame_stop])

    if not os.path.exists(args.frame_folder):
        os.makedirs(args.frame_folder)

    vmin = np.min(logMUA_array)
    vmax = np.max(logMUA_array)

    for num, img in enumerate(logMUA_array):
        fig, ax = plt.subplots()
        img = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray,
                        vmin=vmin, vmax=vmax)
        # fig.colorbar(img)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('pixel size: {} mm'.format(args.pixel_size))
        ax.set_xlabel('{:.2f} {}'.format(times[num],
                                         times.units.dimensionality.string))
        plt.tight_layout
        plt.savefig(os.path.join(args.frame_folder,
                                 args.frame_name
                                 + '_{}.{}'.format(str(num).zfill(5),
                                                   args.frame_format)))
        plt.close(fig)
