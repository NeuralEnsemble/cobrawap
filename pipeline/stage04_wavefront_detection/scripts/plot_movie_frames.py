import os
import sys
import argparse
import matplotlib.pyplot as plt
import neo
import numpy as np
import scipy
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import AnalogSignal2ImageSequence

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--frame_folder",  nargs='?', type=str)
    CLI.add_argument("--frame_name",    nargs='?', type=str)
    CLI.add_argument("--frame_format",  nargs='?', type=str)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        blk = io.read_block()

    blk = AnalogSignal2ImageSequence(blk)

    events = blk.segments[0].events
    imgseq = blk.segments[0].imagesequences[0]
    asig = blk.segments[0].analogsignals[0]

    event = [ev for ev in events if ev.name == 'Transitions'][0]
    ups = np.array([(t,event.array_annotations['channels'][i])
                     for i, t in enumerate(event)
                     if event.labels[i].decode('UTF-8') == 'UP'],
                   dtype=[('time', 'float'), ('channel', 'int')])
    ups = np.sort(ups, order=['time', 'channel'])

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

    for num, frame in enumerate(imgseq.as_array()):
        fig, ax = plt.subplots()
        img = ax.imshow(frame, interpolation='nearest', cmap=plt.cm.gray,
                        vmin=vmin, vmax=vmax)

        if len(ups) and up_coords[num].size:
            ax.plot(up_coords[num][:,1], up_coords[num][:,0],
                    marker='D', color='b', markersize=1, linestyle='None')
            # if len(pixels[0]) > 0.005*pixel_num:
            #     slope, intercept, _, _, stderr = scipy.stats.linregress(pixels[1], pixels[0])
            #     if stderr < 0.18:
            #         ax.plot(x, [intercept + slope*xi for xi in x], color='r')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0, dim_x))
        ax.set_ylim((dim_y, 0))
        ax.set_ylabel('pixel size: {}'\
                      .format(asig.annotations['spatial_scale']))
        ax.set_xlabel('{:.2f} {}'.format(asig.times[num],
                                    asig.times.units.dimensionality.string))

        if not os.path.exists(args.frame_folder):
            os.makedirs(args.frame_folder)
        plt.savefig(os.path.join(args.frame_folder,
                                 args.frame_name
                                 + '_{}.{}'.format(str(num).zfill(5),
                                                   args.frame_format)))
        plt.close(fig)
