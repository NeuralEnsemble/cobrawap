import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import argparse
import scipy
import pandas as pd
import seaborn as sns
from utils import load_neo, save_plot, none_or_str

def calc_displacement(times, locations):
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    d0, d1 = offset + slope*times[0], offset + slope*times[-1]
    displacement = d1 - d0
    displacement_err = np.sqrt(stderr**2 + (stderr*(times[-1]-times[0]))**2)
    return displacement, displacement_err

def calc_directions(evts):
    spatial_scale = evts.annotations['spatial_scale']

    wave_ids = np.unique(evts.labels)

    directions = np.zeros((len(wave_ids), 2), dtype=np.complex_)

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        # Fit wave displacement
        idx = np.where(evts.labels == wave_i)[0]
        dx, dx_err = calc_displacement(evts.times[idx].magnitude,
                                   evts.array_annotations['x_coords'][idx]
                                 * spatial_scale.magnitude)
        dy, dy_err = calc_displacement(evts.times[idx].magnitude,
                                   evts.array_annotations['y_coords'][idx]
                                 * spatial_scale.magnitude)
        directions[i] = np.array([dx + 1j*dy, dx_err + 1j*dy_err])

    ncols = int(np.round(np.sqrt(len(wave_ids))))
    nrows = int(np.ceil((len(wave_ids))/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(3*nrows, 3*ncols))

    rmax = np.max(np.abs(directions[:,0]))
    for i, d in enumerate(directions):
        row = int(i/ncols)
        if ncols == 1:
            cax = ax[row]
        else:
            col = i % ncols
            cax = ax[row][col]

        cax.plot([0,np.imag(d[0])], [0,np.real(d[0])], color='r', alpha=0.8)
        ellipsis = patches.Ellipse(xy=(np.imag(d[0]), np.real(d[0])),
                                   width=2*np.real(d[1]), height=2*np.imag(d[1]),
                                   alpha=0.5)
        cax.add_artist(ellipsis)
        cax.set_title('wave {}'.format(wave_ids[i]))
        if np.isfinite(rmax):
            cax.set_ylim((-rmax,rmax))
            cax.set_xlim((-rmax,rmax))
        cax.axhline(0, c='k')
        cax.axvline(0, c='k')
        cax.axes.get_xaxis().set_visible(False)
        cax.axes.get_yaxis().set_visible(False)
        cax.set_xticks([])
        cax.set_yticks([])
        if not i:
            if 'orientation_right' in evts.annotations:
                cax.set_xlabel(evts.annotations['orientation_right'])
            if 'orientation_top' in evts.annotations:
            cax.set_ylabel(evts.annotations['orientation_top'])
        sns.despine(left=True, bottom=True)


    for i in range(len(directions), nrows*ncols):
        row = int(i/ncols)
        col = i % ncols
        ax[row][col].set_axis_off()

    # transfrom to DataFrame
    df = pd.DataFrame(directions,
                      columns=['direction', 'direction_std'],
                      index=wave_ids)
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

    directions_df = calc_directions(evts)

    if args.output_img is not None:
        save_plot(args.output_img)

    directions_df.to_csv(args.output)
