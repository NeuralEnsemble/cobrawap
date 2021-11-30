"""
Docstring
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from utils import load_neo, save_plot, none_or_str
from scipy.ndimage import convolve as conv
from utils import AnalogSignal2ImageSequence

# # Simple kernel:
# kernelX = np.array([[-0, 0, 0],
#                     [-1, 0, 1],
#                     [-0, 0, 0]], dtype=float) * 1/2
# kernelY = np.array([[-0, -1, -0],
#                     [ 0,  0,  0],
#                     [ 0,  1,  0]], dtype=float) * 1/2
# center = (1,1)

# Sobol kernel:
kernelX = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float) * 1/8
kernelY = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=float) * 1/8
center = (1,1)

# # Large kernel
# kernelX = np.array([[-1.1, -1.3, 0, 1.3, 1.1],
#                     [-1.3, -2.1, 0, 2.1, 1.3],
#                     [-1.5, -3.0, 0, 3.0, 1.5],
#                     [-1.3, -2.1, 0, 2.1, 1.3],
#                     [-1.1, -1.3, 0, 1.3, 1.1]], dtype=float) * 1/32.2
# kernelY = np.array([[-1.1, -1.3, -1.5, -1.3, -1.1],
#                     [-1.3, -2.1, -3.0, -2.1, -1.3],
#                     [ 0,    0,    0,    0,    0],
#                     [ 1.3,  2.1,  3.0,  2.1, 1.3],
#                     [ 1.1,  1.3,  1.5,  1.3, 1.1]], dtype=float) * 1/32.2
# center = (2,2)

def nanconv2d(frame, kernel, kernel_center=None):
    dx, dy = kernel.shape
    dimx, dimy = frame.shape
    dframe = np.empty((dimx, dimy))*np.nan

    if kernel_center is None:
        kernel_center = [int((dim-1)/2) for dim in kernel.shape]

    # inverse kernel to mimic behavior or regular convolution algorithm
    k = kernel[::-1, ::-1]
    ci = dx - 1 - kernel_center[0]
    cj = dy - 1 - kernel_center[1]

    # loop over each frame site
    for i,j in zip(*np.where(np.isfinite(frame))):
        site = frame[i,j]

        # loop over kernel window for frame site
        window = np.zeros((dx,dy), dtype=float)*np.nan
        for di,dj in itertools.product(range(dx), range(dy)):

            # kernelsite != 0, framesite within borders and != nan
            if k[di,dj] and 0 <= i+di-ci < dimx and 0 <= j+dj-cj < dimy \
                        and np.isfinite(frame[i+di-ci,j+dj-cj]):
                sign = -1*np.sign(k[di,dj])
                window[di,dj] = sign * (site - frame[i+di-ci,j+dj-cj])

        xi, yi = np.where(np.logical_not(np.isnan(window)))
        if np.sum(np.logical_not(np.isnan(window))) > dx*dy/10:
            dframe[i,j] = np.average(window[xi,yi], weights=abs(k[xi,yi]))
    return dframe


def calc_local_velocities(wave_evts, dim_x, dim_y):
    evts = wave_evts[wave_evts.labels != '-1']
    labels = evts.labels.astype(int)

    scale = evts.annotations['spatial_scale'].magnitude
    unit = evts.annotations['spatial_scale'].units / evts.times.units

    channel_ids = np.empty([dim_x, dim_y]) * np.nan
    channel_ids[evts.array_annotations['x_coords'].astype(int),
                evts.array_annotations['y_coords'].astype(int)] = evts.array_annotations['channels']
    channel_ids = channel_ids.reshape(-1)

    velocities = np.array([], dtype=float)
    wave_ids = np.array([], dtype=int)
    channels = np.array([], dtype=int)

    for wave_id in np.unique(labels):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)

        trigger_collection = np.empty([dim_x, dim_y]) * np.nan
        trigger_collection[x_coords, y_coords] = wave_trigger_evts.times

        # ToDo: use derivate kernel convolution instead (while ignoring nans) [done]
        # Tx = np.diff(trigger_collection, axis=0, append=np.nan).reshape(-1)
        # Ty = np.diff(trigger_collection, axis=1, append=np.nan).reshape(-1)
        t_x = nanconv2d(trigger_collection, kernelX).reshape(-1)
        t_y = nanconv2d(trigger_collection, kernelY).reshape(-1)

        # polar
        v = np.sqrt((2*scale**2)/(t_x**2 + t_y**2))
        # # cartesian
        # v = np.sqrt((scale/t_x)**2 + (scale/t_y)**2)
        channel_idx = np.where(np.isfinite(v))[0]

        velocities = np.append(velocities, v[channel_idx])
        channels = np.append(channels, channel_ids[channel_idx])
        wave_ids = np.append(wave_ids, np.repeat(wave_id, len(channel_idx)))

    return wave_ids, channels, velocities*unit


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")

    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)

    imgseq = block.segments[0].imagesequences[0]
    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name='Wavefronts', objects="Event")[0]

    dim_t, dim_x, dim_y = np.shape(imgseq)
    wave_ids, channel_ids, velocities = calc_local_velocities(evts, dim_x, dim_y)


    # transform to DataFrame
    df = pd.DataFrame(list(zip(wave_ids, velocities.magnitude)),
                      columns=['wave_id', 'velocity_local'],
                      index=channel_ids)
    df['velocity_local_unit'] = [velocities.dimensionality.string]*len(channel_ids)
    df.index.name = 'channel_id'

    df.to_csv(args.output)

    plt.subplots()
    if args.output_img is not None:
        save_plot(args.output_img)
