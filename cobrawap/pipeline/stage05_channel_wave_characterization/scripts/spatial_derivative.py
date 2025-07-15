"""
Calculate the spatial derivative on the time-delays of the wave triggers
in each channel.

The derivative is calculated using a kernel convolution.
"""

import argparse
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from utils.convolve import get_kernel, nan_conv2d
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_path, none_or_str, str_to_bool

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_path, default=None,
                 help="path of output image file")
CLI.add_argument("--kernel", "--KERNEL", nargs='?', type=none_or_str, default=None,
                 help="derivative kernel")
CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                 help="name of neo.Event to analyze (must contain waves)")
CLI.add_argument("--interpolate", "--INTERPOLATE", nargs='?', type=str_to_bool, default=False,
                 help="whether to thin-plate-spline interpolate the wave patterns before derivation")
CLI.add_argument("--smoothing", "--SMOOTHING", nargs='?', type=float, default=0,
                 help="smoothing factor for the interpolation")

def interpolate_grid(grid, smoothing):
    y, x = np.where(np.isfinite(grid))
    rbf_func = RBFInterpolator(np.stack((y,x), axis=-1),
                               grid[y,x],
                               neighbors=None,
                               smoothing=smoothing,
                               kernel='thin_plate_spline',
                               epsilon=None,
                               degree=None)
    return rbf_func

def sample_wave_pattern(pattern_func, dim_x, dim_y):
    fy, fx = np.meshgrid(np.arange(0, dim_y),
                         np.arange(0, dim_x),
                         indexing='ij')
    fcoords = np.stack((fy,fx), axis=-1)
    wave_pattern = pattern_func(fcoords.reshape(-1,2))
    return wave_pattern.reshape(dim_y, dim_x)

def calc_spatial_derivative(evts, kernel_name, interpolate=False, smoothing=0):
    labels = evts.labels.astype(int)
    dim_x = int(max(evts.array_annotations['x_coords']))+1
    dim_y = int(max(evts.array_annotations['y_coords']))+1

    spatial_derivative_df = pd.DataFrame()

    for wave_id in np.unique(labels):
        wave_trigger_evts = evts[labels == wave_id]

        x_coords = wave_trigger_evts.array_annotations['x_coords'].astype(int)
        y_coords = wave_trigger_evts.array_annotations['y_coords'].astype(int)
        channels = wave_trigger_evts.array_annotations['channels'].astype(int)

        trigger_collection = np.empty([dim_y, dim_x]) * np.nan
        trigger_collection[y_coords, x_coords] = wave_trigger_evts.times

        if interpolate:
            try:
                pattern_func = interpolate_grid(trigger_collection, smoothing)
                trigger_collection = sample_wave_pattern(pattern_func,
                                                         dim_x=dim_x, dim_y=dim_y)
            except ValueError as ve:
                warn(repr(ve))
                warn('Continuing without interpolation.')

        kernel = get_kernel(kernel_name)
        d_horizont = -1 * nan_conv2d(trigger_collection, kernel.x) # is -1 correct?
        d_vertical = -1 * nan_conv2d(trigger_collection, kernel.y)

        dt_y = d_vertical[y_coords, x_coords]
        dt_x = d_horizont[y_coords, x_coords]

        df = pd.DataFrame(list(zip(dt_x, dt_y, x_coords, y_coords, channels)),
                          columns=['dt_x', 'dt_y', 'x_coords', 'y_coords', 'channel_id'])
        df[f'{evts.name}_id'] = wave_id

        spatial_derivative_df = pd.concat([spatial_derivative_df, df],
                                          ignore_index=True)

    fig, ax = plt.subplots(ncols=3, figsize=(15,5))
    img = ax[0].imshow(trigger_collection, cmap='viridis', origin='lower')
    plt.colorbar(img, ax=ax[0])
    vminmax = np.nanmax(abs(d_vertical))
    img = ax[1].imshow(d_vertical, origin='lower', cmap='coolwarm',
                       vmin=-vminmax, vmax=vminmax)
    plt.colorbar(img, ax=ax[1])
    vminmax = np.nanmax(abs(d_horizont))
    img = ax[2].imshow(d_horizont, origin='lower', cmap='coolwarm',
                       vmin=-vminmax, vmax=vminmax)
    plt.colorbar(img, ax=ax[2])
    ax[0].set_title(f'wave {wave_id}')
    ax[1].set_title('dt Y (vertical)')
    ax[2].set_title('dt X (horizontal)')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()

    return spatial_derivative_df


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels != '-1']

    df = calc_spatial_derivative(evts,
                                 kernel_name=args.kernel,
                                 interpolate=args.interpolate,
                                 smoothing=args.smoothing)

    df['kernel'] = args.kernel
    df['spatial_scale'] = evts.annotations['spatial_scale'].magnitude
    df['spatial_scale_unit'] = evts.annotations['spatial_scale'].dimensionality.string
    df['dt_unit'] = evts.times.dimensionality.string

    df.to_csv(args.output)

    if args.output_img is not None:
        save_plot(args.output_img)
