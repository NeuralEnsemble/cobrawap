import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import pandas as pd
from utils import load_neo, none_or_str, save_plot

def center_points(x, y):
    return x - np.mean(x), y - np.mean(y)

def linregress(times, locations):
    times, locations = center_points(times, locations)
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    return slope, stderr, offset


def calc_planar_velocities(evts):
    spatial_scale = evts.annotations['spatial_scale']
    v_unit = (spatial_scale.units/evts.times.units).dimensionality.string

    wave_ids = np.unique(evts.labels)

    velocities = np.zeros((len(wave_ids), 2))

    ncols = int(np.round(np.sqrt(len(wave_ids)+1)))
    nrows = int(np.ceil((len(wave_ids)+1)/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(3*nrows, 3*ncols))

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        # Fit wave displacement
        idx = np.where(evts.labels == wave_i)[0]
        x_times, x_locations = center_points(evts.times[idx].magnitude,
                                        evts.array_annotations['x_coords'][idx]
                                        * spatial_scale.magnitude)
        y_times, y_locations = center_points(evts.times[idx].magnitude,
                                        evts.array_annotations['y_coords'][idx]
                                        * spatial_scale.magnitude)
        vx, vx_err, dx = linregress(x_times, x_locations)
        vy, vy_err, dy = linregress(y_times, y_locations)
        v = np.sqrt(vx**2 + vy**2)
        print(np.concatenate((x_times[:, np.newaxis],
                              x_locations[:, np.newaxis],
                              y_locations[:, np.newaxis]), axis=1))
        v_err = 1/v * np.sqrt((vx*vx_err)**2 + (vy+vy_err)**2)
        velocities[i] = np.array([v, v_err])

        # Plot fit
        row = int(i/ncols)
        if ncols == 1:
            cax = ax[row]
        else:
            col = i % ncols
            cax = ax[row][col]
        cax.plot(x_times, x_locations,
                color='b', label='x coords', linestyle='', marker='.', alpha=0.5)
        cax.plot(x_times, [vx*t + dx for t in x_times], color='b')
        cax.plot(y_times, y_locations,
                color='r', label='y coords', linestyle='', marker='.', alpha=0.5)
        cax.plot(y_times, [vy*t + dy for t in y_times], color='r')
        if not col:
            cax.set_ylabel('x/y position [{}]'\
                           .format(spatial_scale.dimensionality.string))
        if row == nrows-1:
            cax.set_xlabel('time [{}]'\
                           .format(evts.times[idx].dimensionality.string))
        cax.set_title('wave {}'.format(wave_i))

    # plot total velocities
    ax[-1][-1].errorbar(wave_ids, velocities[:,0], yerr=velocities[:,1],
                        linestyle='', marker='+')
    ax[-1][-1].set_xlabel('wave id')
    ax[-1][-1].set_title('velocities [{}]'.format(v_unit))

    for i in range(len(wave_ids), nrows*ncols-1):
        row = int(i/ncols)
        col = i % ncols
        ax[row][col].set_axis_off()

    # transform to DataFrame
    df = pd.DataFrame(velocities,
                      columns=['velocity_planar', 'velocity_planar_std'],
                      index=wave_ids)
    df['velocity_unit'] = [v_unit]*len(wave_ids)
    df.index.name = 'wave_id'

    return df


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    args = CLI.parse_args()

    block = load_neo(args.data)

    evts = [ev for ev in block.segments[0].events if ev.name == 'Wavefronts'][0]

    velocities_df = calc_planar_velocities(evts)

    if args.output_img is not None:
        save_plot(args.output_img)

    velocities_df.to_csv(args.output)
