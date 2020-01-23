import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import pandas as pd

def calc_velocity(times, locations):
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    return slope, stderr, offset

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
    v_unit = (spatial_scale.units/evts.times.units).dimensionality.string

    wave_ids = [label.decode('UTF-8') for label in np.unique(evts.labels)]

    velocities = np.zeros((len(wave_ids), 2))

    ncols = int(np.round(np.sqrt(len(wave_ids)+1)))
    nrows = int(np.ceil((len(wave_ids)+1)/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(3*nrows, 3*ncols))

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        # Fit wave displacement
        idx = np.where(evts.labels == wave_i.encode('UTF-8'))[0]
        # vx, vx_err, dx = calc_velocity(evts.times[idx].magnitude,
        #                            evts.array_annotations['x_coords'][idx]
        #                            * spatial_scale.magnitude)
        # vy, vy_err, dy = calc_velocity(evts.times[idx].magnitude,
        #                            evts.array_annotations['y_coords'][idx]
        #                            * spatial_scale.magnitude)
        # velocities[i] = np.array([np.sqrt(vx**2 + vy**2),
        #                           np.sqrt(vx_err**2 + vy_err**2)])

        coords = np.array([(evts.array_annotations['x_coords'][i],
                            evts.array_annotations['y_coords'][i])
                            for i in idx])
        values = evts.times[idx].magnitude
        grid_x, grid_y = np.mgrid[0:5:25j, 0:10:50j]

        eps = .1
        grid_z = scipy.interpolate.griddata(coords, values, (grid_x, grid_y), method='cubic')
        grid_z_dx = scipy.interpolate.griddata(coords, values, (grid_x+eps, grid_y), method='cubic')
        grid_z_dy = scipy.interpolate.griddata(coords, values, (grid_x, grid_y+eps), method='cubic')

        dTdx = (grid_z - grid_z_dx)/eps
        dTdy = (grid_z - grid_z_dy)/eps
        v_grad = 1/(np.sqrt(dTdx**2 + dTdy**2)) * spatial_scale.magnitude
        # print(np.nanmean(v_grad))


        # ucoords, uidx = np.unique(coords, axis=0, return_index=True)
        # rbfi = scipy.interpolate.Rbf(ucoords[:,0], ucoords[:,1], values[uidx],
        #                              function='thin_plate', smooth=0)
        # rbfi = scipy.interpolate.Rbf(coords[:,0], coords[:,1], values,
        #                               function='thin_plate', smooth=0)
        # interpolated_data = rbfi(ucoords[:,0], ucoords[:,1])
        # interpolated_data_dx = rbfi(ucoords[:,0]+eps, ucoords[:,1])
        # interpolated_data_dy = rbfi(ucoords[:,0], ucoords[:,1]+eps)
        #
        # interpolated_data = rbfi(grid_x, grid_y)
        # interpolated_data_dx = rbfi(grid_x+eps, grid_y)
        # interpolated_data_dy = rbfi(grid_x, grid_y+eps)
        #
        # dTdy = (interpolated_data_dy - interpolated_data)/eps
        # dTdx = (interpolated_data_dx - interpolated_data)/eps

        # dTdx, dTdy = np.gradient(interpolated_data)

        # v_grad = 1/(np.sqrt(dTdx**2 + dTdy**2)) * spatial_scale.magnitude

        print(wave_i, np.nanmedian(v_grad))

        # Plot fit
        row = int(i/ncols)
        col = i % ncols
        cax = ax[row][col]
        im = cax.imshow(v_grad, extent=(0,1,0,1), origin='lower')
        fig.colorbar(im, ax=cax)

        # cax.plot(evts.times[idx].magnitude,
        #         evts.array_annotations['x_coords'][idx]*spatial_scale.magnitude,
        #         color='b', label='x coords', linestyle='', marker='.', alpha=0.5)
        # cax.plot(evts.times[idx].magnitude,
        #         [vx*t + dx for t in evts.times[idx].magnitude], color='b')
        # cax.plot(evts.times[idx].magnitude,
        #         evts.array_annotations['y_coords'][idx]*spatial_scale.magnitude,
        #         color='r', label='y coords', linestyle='', marker='.', alpha=0.5)
        # cax.plot(evts.times[idx].magnitude,
        #         [vy*t + dy for t in evts.times[idx].magnitude], color='r')
        # if not col:
        #     cax.set_ylabel('x/y position [{}]'\
        #                    .format(spatial_scale.dimensionality.string))
        # if row == nrows-1:
        #     cax.set_xlabel('time [{}]'\
        #                    .format(evts.times[idx].dimensionality.string))
        # cax.set_title('wave {}'.format(wave_i))

    # plot total velocities
    # ax[-1][-1].errorbar(wave_ids, velocities[:,0], yerr=velocities[:,1],
    #                     linestyle='', marker='+')
    ax[-1][-1].set_xlabel('wave id')
    ax[-1][-1].set_title('velocities [{}]'.format(v_unit))
    #
    # for i in range(len(wave_ids), nrows*ncols-1):
    #     row = int(i/ncols)
    #     col = i % ncols
    #     ax[row][col].set_axis_off()
    #
    plt.tight_layout()
    plt.show()
    # plt.savefig(args.output_img)

    # save as DataFrame
    df = pd.DataFrame(velocities,
                      columns=['velocity', 'velocity_std'],
                      index=wave_ids)
    df['velocity_unit'] = [v_unit]*len(wave_ids)
    df.index.name = 'wave_id'

    df.to_csv(args.output)
