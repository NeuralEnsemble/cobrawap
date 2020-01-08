import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import pandas as pd

def calc_displacement(times, locations):
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    # ToDo
    d0, d1 = offset + slope*times[0], offset + slope*times[-1]
    displacement = d1 - d0
    displacement_err = np.sqrt((stderr*times[0])**2 + (stderr*times[-1])**2)
    return displacement, displacement_err

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--output_img",      nargs='?', type=str)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    evts = [ev for ev in block.segments[0].events if ev.name== 'Wavefronts'][0]

    spatial_scale = evts.annotations['spatial_scale']
    # v_unit = (spatial_scale.units/evts.times.units).dimensionality.string

    wave_ids = [label.decode('UTF-8') for label in np.unique(evts.labels)]

    directions = np.zeros((len(wave_ids), 2), dtype=np.complex_)

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        # Fit wave displacement
        idx = np.where(evts.labels == wave_i.encode('UTF-8'))[0]
        dx, dx_err = calc_displacement(evts.times[idx].magnitude,
                                   evts.array_annotations['x_coords'][idx]
                                 * spatial_scale.magnitude)
        dy, dy_err = calc_displacement(evts.times[idx].magnitude,
                                   evts.array_annotations['y_coords'][idx]
                                 * spatial_scale.magnitude)
        directions[i] = np.array([dx + 1j*dy, dx_err - 1j*dy_err])

    nrows = int(np.round(np.sqrt(len(wave_ids))))
    ncols = int(np.ceil((len(wave_ids))/nrows))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           subplot_kw={'projection': 'polar'})

    rmax = np.max(np.abs(directions[:,0]))
    for i, d in enumerate(directions):
        row = int(i/nrows)
        col = i-row*ncols
        cax = ax[row][col]
        cax.plot([np.angle(d[0])]*2, [0,np.abs(d[0])], color='r', alpha=0.8)
        cax.fill_between(np.linspace(np.angle(d[0]+d[1]), np.angle(d[0]-d[1]), 40),
                         np.linspace(0, 0, 40),
                         np.linspace(np.abs(d[0]), np.abs(d[0]), 40),
                         alpha=0.7)
        cax.set_title('wave {}'.format(wave_ids[i]))
        cax.set_ylim((0,rmax))
        cax.set_xticks([0])
        cax.set_xticklabels(['x'])

    for i in range(len(directions), nrows*ncols):
        row = int(i/nrows)
        col = i-row*ncols
        ax[row][col].set_axis_off()

    plt.tight_layout()
    plt.savefig(args.output_img)

    # save as DataFrame
    df = pd.DataFrame(directions,
                      columns=['direction', 'direction_std'],
                      index=wave_ids)
    # df['direction_unit'] = [v_unit]*len(wave_ids)

    df.to_csv(args.output)
