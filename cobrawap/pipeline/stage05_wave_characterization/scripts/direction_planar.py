"""
Calculate the wave directions by either interpolating trigger times and
locations or by averaging the corresponding optical flow values.
"""

import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import warnings
import argparse
from pathlib import Path
import scipy
import pandas as pd
import seaborn as sns
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image file")
CLI.add_argument("--method", "--DIRECTION_METHOD", nargs='?', type=str, default='trigger_interpolation',
                 help="'tigger_interpolation' or 'optical_flow'")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")

def calc_displacement(times, locations):
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    d0, d1 = offset + slope*times[0], offset + slope*times[-1]
    displacement = d1 - d0
    displacement_err = np.sqrt(stderr**2 + (stderr*(times[-1]-times[0]))**2)
    return displacement, displacement_err

def trigger_interpolation(evts):
    spatial_scale = evts.annotations['spatial_scale']
    wave_ids = np.unique(evts.labels)
    dx_avg, dy_avg = np.zeros(len(wave_ids)), np.zeros(len(wave_ids))
    dx_std, dy_std = np.zeros(len(wave_ids)), np.zeros(len(wave_ids))

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        # Fit wave displacement
        idx = np.where(evts.labels == wave_i)[0]
        dx_avg[i], dx_std[i] = calc_displacement(evts.times[idx].magnitude,
                                   evts.array_annotations['x_coords'][idx]
                                 * spatial_scale.magnitude)
        dy_avg[i], dy_std[i] = calc_displacement(evts.times[idx].magnitude,
                                   evts.array_annotations['y_coords'][idx]
                                 * spatial_scale.magnitude)
    return dx_avg, dy_avg, dx_std, dy_std


def times2ids(time_array, times_selection):
    return np.array([np.argmax(time_array>=t) for t in times_selection])

def calc_flow_direction(evts, asig):
    wave_ids = np.unique(evts.labels.astype(int))
    dx_avg, dy_avg = np.zeros(len(wave_ids)), np.zeros(len(wave_ids))
    dx_std, dy_std = np.zeros(len(wave_ids)), np.zeros(len(wave_ids))
    signals = asig.as_array()

    for i, wave_i in enumerate(wave_ids):
        idx = np.where(evts.labels.astype(int) == wave_i)[0]
        t_idx = times2ids(asig.times, evts.times[idx])
        channels = evts.array_annotations['channels'][idx]
        flow_vectors = signals[t_idx, channels]
        flow_vectors /= np.abs(flow_vectors)
        if np.isnan(flow_vectors).any():
            warnings.warn("Signals at trigger points contain nans!")
        dx_avg[i] = np.nanmean(np.real(flow_vectors))
        dx_std[i] = np.nanstd(np.real(flow_vectors))
        dy_avg[i] = np.nanmean(np.imag(flow_vectors))
        dy_std[i] = np.nanstd(np.imag(flow_vectors))
    return dx_avg, dy_avg, dx_std, dy_std

def plot_directions(dataframe, wave_ids,
                    orientation_top=None, orientation_right=None):
    directions = dataframe.direction_x + dataframe.direction_y*1j
    directions_std = dataframe.direction_x_std + dataframe.direction_y_std*1j

    ncols = int(np.round(np.sqrt(len(wave_ids)+1)))
    nrows = int(np.ceil((len(wave_ids)+1)/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(3*nrows, 3*ncols))

    rmax = np.max(np.abs(directions))
    for i, (d, d_std) in enumerate(zip(directions, directions_std)):
        row = int(i/ncols)
        if ncols == 1:
            cax = ax[row]
        else:
            col = i % ncols
            cax = ax[row][col]

        cax.plot([0,np.real(d)], [0,np.imag(d)], color='r', alpha=0.8)
        ellipsis = patches.Ellipse(xy=(np.real(d), np.imag(d)),
                                   width=2*np.real(d_std), height=2*np.imag(d_std),
                                   alpha=0.5)
        cax.add_artist(ellipsis)
        cax.set_title('wave {}'.format(wave_ids[i]))
        if np.isfinite(rmax):
            cax.set_ylim((-rmax,rmax))
            cax.set_xlim((-rmax,rmax))
        cax.axhline(0, c='k')
        cax.axvline(0, c='k')
        # cax.axes.get_xaxis().set_visible(False)
        # cax.axes.get_yaxis().set_visible(False)
        cax.set_xticks([])
        cax.set_yticks([])

    if ncols == 1:
        cax = ax[-1]
    else:
        cax = ax[-1][-1]
        for i in range(len(directions), nrows*ncols):
            row = int(i/ncols)
            col = i % ncols
            ax[row][col].set_axis_off()

    cax.axhline(0, c='k')
    cax.axvline(0, c='k')
    cax.set_xlim((-2,2))
    cax.set_ylim((-2,2))
    if orientation_top is not None:
        cax.text(0, 1,orientation_top, rotation='vertical',
                        verticalalignment='center', horizontalalignment='right')
    if orientation_right is not None:
        cax.text(1, 0, orientation_right,
                        verticalalignment='top', horizontalalignment='center')

    sns.despine(left=True, bottom=True)
    return ax

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    if args.method == 'optical_flow':
        if args.event_name == 'wavemodes':
            warnings.warn('The planar direction of wavemodes can not be '
                          'calculated with the optical_flow method. '
                          'Using trigger_interpolation instead.')
            args.method = 'trigger_interpolation'
        elif not len(block.filter(name='optical_flow', objects="AnalogSignal")):
            warnings.warn('No optical_flow signal could be found for the '
                          'calculation of planar directions. '
                          'Using trigger_interpolation instead.')
            args.method = 'trigger_interpolation'

    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels.astype('str') != '-1']

    if args.method == 'trigger_interpolation':
        dx_avg, dy_avg, dx_std, dy_std = trigger_interpolation(evts)
    elif args.method == 'optical_flow':
        asig = block.filter(name='optical_flow', objects="AnalogSignal")[0]
        dx_avg, dy_avg, dx_std, dy_std = calc_flow_direction(evts, asig)
    else:
        raise NameError(f'Method name {args.method} is not recognized!')

    df = pd.DataFrame(np.unique(evts.labels.astype(int)),
                      columns=[f'{args.event_name}_id'])
    df['direction_x'] = dx_avg
    df['direction_y'] = dy_avg
    df['direction_x_std'] = dx_std
    df['direction_y_std'] = dy_std

    if args.output_img is not None:
        plot_directions(df,
                        wave_ids=np.unique(evts.labels.astype(int)),
                        orientation_top=evts.annotations['orientation_top'],
                        orientation_right=evts.annotations['orientation_right'])
        save_plot(args.output_img)

    df.to_csv(args.output)
