"""
Create frames showing the input data on the spatial grid for each time step.
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils.io_utils import load_neo, save_plot
from utils.neo_utils import analogsignal_to_imagesequence
from utils.parse import none_or_str, none_or_float


CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--frame_folder", nargs='?', type=Path,
                 help="")
CLI.add_argument("--frame_name", nargs='?', type=str,
                 help="")
CLI.add_argument("--frame_format", nargs='?', type=str,
                 help="")
CLI.add_argument("--frame_rate", nargs='?', type=none_or_float,
                 help="")
CLI.add_argument("--colormap", nargs='?', type=str,
                 help="")
CLI.add_argument("--plot_event", nargs='?', type=none_or_str, default=None,
                 help="")
CLI.add_argument("--marker_color", nargs='?', type=str, default='k',
                 help="")


def get_events(events, frame_times, event_name='transitions'):
    trans_events = [ev for ev in events if ev.name == event_name]
    if len(trans_events):
        event = trans_events[0]
        ups = np.array([(t,
                         event.array_annotations['x_coords'][i],
                         event.array_annotations['y_coords'][i])
                         for i, t in enumerate(event)],
                         # if event.labels[i] == 'UP'],
                       dtype=[('time', 'float'),
                              ('x_coords', 'int'),
                              ('y_coords', 'int')])
        ups = np.sort(ups, order=['time', 'x_coords', 'y_coords'])

        up_coords = []
        for frame_count, frame_time in enumerate(frame_times):
            # select indexes of up events during this frame
            idx = range(np.argmax(np.bitwise_not(ups['time'] < frame_time)),
                        np.argmax(ups['time'] > frame_time))
            frame_ups = np.array([(x,y) for x, y in zip(ups['x_coords'][idx],
                                                        ups['y_coords'][idx])])
            up_coords.append(frame_ups)
        return up_coords
    else:
        print(f"No {event_name} events found!")
        return None


def get_opticalflow(block, imgseq_name="optical_flow"):
    asigs = block.filter(name=imgseq_name, objects="AnalogSignal")
    if len(asigs):
        # Normalize?
        return analogsignal_to_imagesequence(asigs[0])
    else:
        return None

def stretch_to_framerate(t_start, t_stop, num_frames, frame_rate=None):
    if frame_rate is None:
        return np.arange(num_frames, dtype=int)
    else:
        new_num_frames = (t_stop.rescale('s').magnitude
                        - t_start.rescale('s').magnitude) \
                        * frame_rate
        return np.linspace(0, num_frames-1, int(new_num_frames), dtype=int)

def plot_frame(frame, cmap=plt.cm.gray, vmin=None, vmax=None):
    fig, ax = plt.subplots()
    img = ax.imshow(frame, interpolation='nearest',
                    cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(img, pad=0, ax=ax)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlim((0, dim_x))
    # ax.set_ylim((dim_y, 0))
    return ax

def plot_transitions(up_coords, markersize=1, markercolor='k', ax=None):
    if up_coords.size:
        if ax is None:
            ax = plt.gca()

        ax.plot(up_coords[:,0], up_coords[:,1],
                marker='D', color=markercolor, markersize=markersize,
                linestyle='None', alpha=0.6)
        # if len(pixels[0]) > 0.005*pixel_num:
        #     slope, intercept, _, _, stderr = scipy.stats.linregress(pixels[1], pixels[0])
        #     if stderr < 0.18:
        #         ax.plot(x, [intercept + slope*xi for xi in x], color='r')
    return ax

def plot_vectorfield(frame, skip_step=3, ax=None):
    # Every <skip_step> point in each direction.
    if ax is None:
        ax = plt.gca()
    dim_y, dim_x = frame.shape
    y_idx, x_idx = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
    xi = x_idx[::skip_step,::skip_step].flatten()
    yi = y_idx[::skip_step,::skip_step].flatten()
    ax.quiver(xi, yi, np.real(frame[yi,xi]), np.imag(frame[yi,xi]))
    return ax

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    blk = load_neo(args.data)
    asig = blk.segments[0].analogsignals[0]
    imgseq = analogsignal_to_imagesequence(asig)

    optical_flow = get_opticalflow(blk)

    # get data
    times = blk.segments[0].analogsignals[0].times  # to be replaced
    t_start = blk.segments[0].analogsignals[0].t_start  # to be replaced
    t_stop = blk.segments[0].analogsignals[0].t_stop  # to be replaced
    dim_t, dim_y, dim_x = imgseq.shape

    if args.plot_event is not None:
        up_coords = get_events(blk.segments[0].events,
                               frame_times=times,
                               event_name=args.plot_event)

    # prepare plotting
    frame_idx = stretch_to_framerate(t_start=t_start,
                                     t_stop=t_stop,
                                     num_frames=dim_t,
                                     frame_rate=args.frame_rate)
    if args.colormap == 'gray':
        cmap = plt.cm.gray
    else:
        # 'gray', 'viridis' (sequential), 'coolwarm' (diverging), 'twilight' (cyclic)
        cmap = plt.get_cmap(args.colormap)

    frames = imgseq.as_array()
    vmin = np.nanmin(frames)
    vmax = np.nanmax(frames)
    markersize = 50 / max([dim_x, dim_y])
    skip_step = int(min([dim_x, dim_y]) / 50) + 1

    # plot frames
    for i, frame_num in enumerate(frame_idx):
        ax = plot_frame(frames[frame_num], cmap=cmap, vmin=vmin, vmax=vmax)

        if optical_flow is not None:
            plot_vectorfield(optical_flow[frame_num].as_array(),
                             skip_step=skip_step)
        if args.plot_event is not None and up_coords is not None:
            plot_transitions(up_coords[frame_num], markersize=markersize,
                             markercolor=args.marker_color)
        ax.set_ylabel('pixel size: {:.2f} mm'.format(imgseq.spatial_scale.rescale('mm').magnitude))
        ax.set_xlabel('{:.3f} s'.format(times[frame_num].rescale('s').magnitude))

        save_plot(os.path.join(str(args.frame_folder),
                               args.frame_name + '_{}.{}'.format(str(i).zfill(5),
                               args.frame_format)),
                  transparent=True)
        plt.close()
