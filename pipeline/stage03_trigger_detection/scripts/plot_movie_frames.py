import os
import sys
import argparse
import matplotlib.pyplot as plt
import neo
import numpy as np
import scipy
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import AnalogSignal2ImageSequence

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--frame_folder",  nargs='?', type=str)
    CLI.add_argument("--frame_name",    nargs='?', type=str)
    CLI.add_argument("--frame_format",  nargs='?', type=str)
    CLI.add_argument("--frame_rate",    nargs='?', type=none_or_float)
    CLI.add_argument("--colormap",      nargs='?', type=str)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        blk = io.read_block()

    blk = AnalogSignal2ImageSequence(blk)

    events = blk.segments[0].events
    imgseq = blk.segments[0].imagesequences[0]
    asig = blk.segments[0].analogsignals[0]

    trans_events = [ev for ev in events if ev.name == 'Transitions']
    if len(trans_events):
        event = trans_events[0]
        ups = np.array([(t,event.array_annotations['channels'][i])
                         for i, t in enumerate(event)
                         if event.labels[i].decode('UTF-8') == 'UP'],
                       dtype=[('time', 'float'), ('channel', 'int')])
        ups = np.sort(ups, order=['time', 'channel'])
    else:
        ups = []

    def channels2coords(channels):
        if isinstance(channels, (list, np.ndarray)):
            channels = [channels]
        return np.array([(asig.array_annotations['x_coords'][i],
                          asig.array_annotations['y_coords'][i])
                         for i in channels])

    if len(ups):
        up_coords = []
        for frame_count, frame_time in enumerate(asig.times):
            # select indexes of up events during this frame
            idx = range(np.argmax(np.bitwise_not(ups['time'] < frame_time)),
                        np.argmax(ups['time'] > frame_time))
            up_coords.append(channels2coords(ups['channel'][idx]))

    vmin = np.nanmin(asig.as_array())
    vmax = np.nanmax(asig.as_array())

    dim_x = np.max(asig.array_annotations['x_coords'])
    dim_y = np.max(asig.array_annotations['y_coords'])
    dim_t = len(imgseq.as_array())
    markersize = 100/max([dim_x, dim_y])

    # 'gray', 'viridis' (sequential), 'coolwarm' (diverging), 'twilight' (cyclic)
    if args.colormap == 'gray':
        cmap = plt.cm.gray
    else:
        cmap = plt.get_cmap(args.colormap)

    # strech or compress
    if args.frame_rate is None:
        frame_idx = np.arange(dim_t, dtype=int)
    else:
        num_frames = (asig.t_stop.rescale('s').magnitude
                    - asig.t_start.rescale('s').magnitude) * args.frame_rate
        frame_idx = np.linspace(0, dim_t-1, num_frames).astype(int)

    for i, frame_num in enumerate(frame_idx):
        fig, ax = plt.subplots()
        img = ax.imshow(imgseq.as_array()[frame_num], interpolation='nearest',
                        cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(img, ax=ax)

        if len(ups) and up_coords[frame_num].size:
            ax.plot(up_coords[frame_num][:,1], up_coords[frame_num][:,0],
                    marker='D', color='b', markersize=markersize, linestyle='None')
            # if len(pixels[0]) > 0.005*pixel_num:
            #     slope, intercept, _, _, stderr = scipy.stats.linregress(pixels[1], pixels[0])
            #     if stderr < 0.18:
            #         ax.plot(x, [intercept + slope*xi for xi in x], color='r')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xlim((0, dim_x))
        # ax.set_ylim((dim_y, 0))
        ax.set_ylabel('pixel size: {}'\
                      .format(asig.annotations['spatial_scale']))
        ax.set_xlabel('{:.3f} s'.format(asig.times[frame_num].rescale('s')))

        if not os.path.exists(args.frame_folder):
            os.makedirs(args.frame_folder)
        plt.savefig(os.path.join(args.frame_folder,
                                 args.frame_name
                                 + '_{}{}'.format(str(i).zfill(5),
                                                   args.frame_format)))
        plt.close(fig)
