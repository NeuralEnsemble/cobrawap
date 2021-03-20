import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import warnings
import argparse
import scipy
import pandas as pd
import seaborn as sns
from utils import load_neo, save_plot, none_or_str, AnalogSignal2ImageSequence

def calc_displacement(times, locations):
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    d0, d1 = offset + slope*times[0], offset + slope*times[-1]
    displacement = d1 - d0
    displacement_err = np.sqrt(stderr**2 + (stderr*(times[-1]-times[0]))**2)
    return displacement, displacement_err

def trigger_interpolation(evts):
    spatial_scale = evts.annotations['spatial_scale']

    wave_ids = np.unique(evts.labels)

    directions = np.zeros((len(wave_ids), 2), dtype=np.complex_)

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        if not int(wave_i) == -1:
            # Fit wave displacement
            idx = np.where(evts.labels == wave_i)[0]
            dx, dx_err = calc_displacement(evts.times[idx].magnitude,
                                       evts.array_annotations['x_coords'][idx]
                                     * spatial_scale.magnitude)
            dy, dy_err = calc_displacement(evts.times[idx].magnitude,
                                       evts.array_annotations['y_coords'][idx]
                                     * spatial_scale.magnitude)
            directions[i] = np.array([dx + 1j*dy, dx_err + 1j*dy_err])

    return directions


def times2ids(time_array, times_selection):
    return np.array([np.argmax(time_array>=t) for t in times_selection])

def calc_flow_direction(evts, asig):
    wave_ids = np.unique(evts.labels)
    directions = np.zeros((len(wave_ids), 2), dtype=np.complex_)
    signals = asig.as_array()

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        if not int(wave_i) == -1:
            idx = np.where(evts.labels == str(wave_i))[0]
            t_idx = times2ids(asig.times, evts.times[idx])
            x_coords = evts.array_annotations['x_coords'][idx]
            y_coords = evts.array_annotations['y_coords'][idx]
            channels = np.empty(len(idx), dtype=int)
            for c, (x,y) in enumerate(zip(x_coords, y_coords)):
                channels[c] = np.where((asig.array_annotations['x_coords'] == x) \
                                     & (asig.array_annotations['y_coords'] == y))[0]
            # channels = evts.array_annotations['channels'][idx]
            # ToDo: Normalize vectors?
            if np.isnan(signals[t_idx, channels]).any():
                warnings.warn("Signals at trigger points contain nans!")
            x_avg = np.nanmean(np.real(signals[t_idx, channels]))
            x_std = np.nanstd(np.real(signals[t_idx, channels]))
            y_avg = np.nanmean(np.imag(signals[t_idx, channels]))
            y_std = np.nanstd(np.imag(signals[t_idx, channels]))
            directions[i] = np.array([x_avg + 1j*y_avg, x_std + 1j*y_std])
    return directions

def plot_directions(dataframe, orientation_top=None, orientation_right=None):
    wave_ids = dataframe.index
    directions = dataframe.direction
    directions_std = dataframe.direction_std

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

    ax[-1][-1].axhline(0, c='k')
    ax[-1][-1].axvline(0, c='k')
    ax[-1][-1].set_xlim((-2,2))
    ax[-1][-1].set_ylim((-2,2))
    if orientation_top is not None:
        ax[-1][-1].text(0, 1,orientation_top, rotation='vertical',
                        verticalalignment='center', horizontalalignment='right')
    if orientation_right is not None:
        ax[-1][-1].text(1, 0, orientation_right,
                        verticalalignment='top', horizontalalignment='center')

    sns.despine(left=True, bottom=True)

    for i in range(len(directions), nrows*ncols):
        row = int(i/ncols)
        col = i % ncols
        ax[row][col].set_axis_off()
    return ax

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--method", nargs='?', type=str, default='trigger_interpolation',
                     help="'tigger_interpolation' or 'optical_flow'")
    args = CLI.parse_args()

    block = load_neo(args.data)

    evts = block.filter(name='Wavefronts', objects="Event")[0]

    if args.method == 'trigger_interpolation':
        directions = trigger_interpolation(evts)
    elif args.method == 'optical_flow':
        asig = block.filter(name='Optical Flow', objects="AnalogSignal")[0]
        directions = calc_flow_direction(evts, asig)
    else:
        raise NameError(f'Method name {args.method} is not recognized!')

    df = pd.DataFrame(directions,
                      columns=['direction', 'direction_std'],
                      index=np.unique(evts.labels))
    df.index.name = 'wave_id'

    if args.output_img is not None:
        orientation_top = evts.annotations['orientation_top']
        orientation_right = evts.annotations['orientation_right']
        plot_directions(df, orientation_top, orientation_right)
        save_plot(args.output_img)

    df.to_csv(args.output)
