import os
import sys
import argparse
import matplotlib.pyplot as plt
import imageio
import subprocess
import neo
import numpy as np
import scipy
sys.path.append(os.getcwd())


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file",    nargs='?', type=str)
    CLI.add_argument("--frame_folder",  nargs='?', type=str)
    CLI.add_argument("--frame_name",    nargs='?', type=str)
    CLI.add_argument("--frame_format",  nargs='?', type=str)

    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        seg = io.read_block().segments[0]
        images = seg.analogsignals[0]
        up_trains = seg.spiketrains

    if not os.path.exists(args.frame_folder):
        os.makedirs(args.frame_folder)

    if len(up_trains):
        # for every frame, list of up coords
        up_coords = [[] for t in images.times]
        for up_train in up_trains:
            for up_time in up_train:
                t_idx = np.where(images.times >= up_time)[0][0]
                up_coords[t_idx].append(up_train.annotations['coordinates'])

    vmin = np.nanmin(images.as_array())
    vmax = np.nanmax(images.as_array())

    pixel_num = (~np.isnan(images[0])).sum()
    dim_x, dim_y = images[0].shape
    x = np.arange(dim_x)


    for num, img in enumerate(images):
        fig, ax = plt.subplots()
        img = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray,
                        vmin=vmin, vmax=vmax)

        if len(up_trains) and len(up_coords[num]):
            pixels = np.array(up_coords[num]).T
            ax.plot(pixels[1], pixels[0], marker='D', color='b', markersize=1, linestyle='None')
            if len(pixels[0]) > 0.005*pixel_num:
                slope, intercept, _, _, stderr = scipy.stats.linregress(pixels[1], pixels[0])
                if stderr < 0.18:
                    ax.plot(x, [intercept + slope*xi for xi in x], color='r')

        # fig.colorbar(img)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0, dim_x))
        ax.set_ylim((dim_y, 0))
        ax.set_ylabel('pixel size: {} mm'\
                      .format(images.annotations['pixel_size']))
        ax.set_xlabel('{:.2f} {}'.format(images.times[num],
                                    images.times.units.dimensionality.string))


        plt.savefig(os.path.join(args.frame_folder,
                                 args.frame_name
                                 + '_{}.{}'.format(str(num).zfill(5),
                                                   args.frame_format)))
        plt.close(fig)
