"""
Plot snapshots of the input data showing the detected waves.
"""

import os
import numpy as np
import quantities as pq
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, save_plot
from utils.neo_utils import analogsignal_to_imagesequence

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output_dir", nargs='?', type=Path, required=True,
                 help="path to output directory")
CLI.add_argument("--img_name", nargs='?', type=str,
                 help="")
CLI.add_argument("--time_window", nargs='?', type=float, default=0.4,
                 help="size of the plotted window in seconds.")
CLI.add_argument("--colormap", nargs='?', type=str, default='viridis',
                 help="")

def plot_wave(wave_id, waves_event, asig, frames, vec_frames,
              time_window=0.4*pq.s, cmap='virids'):
    idx = np.where(waves_event.labels == str(wave_id))[0]
    x_coords = waves_event.array_annotations['x_coords'][idx]
    y_coords = waves_event.array_annotations['y_coords'][idx]

    t = waves_event.times[idx]
    time_steps = np.unique(t)

    dim_t, dim_y, dim_x = frames.shape
    y_idx, x_idx = np.meshgrid(np.arange(dim_y), np.arange(dim_x), indexing='ij')
    markersize = 50 / max([dim_x, dim_y])
    skip_step = int(min([dim_x, dim_y]) / 50) + 1

    vmin, vmax = np.nanmin(frames), np.nanmax(frames)

    time_window_steps = asig.sampling_rate.rescale('Hz') * time_window.rescale('s')
    half_window = int(time_window_steps.magnitude / 2)

    fig, axes = plt.subplots(nrows=2, ncols=len(time_steps),
                             figsize=(2*len(time_steps), 5), sharey='row')
    if len(time_steps) < 2:
        axes = [[axes[0]], [axes[1]]]

    axes[0][0].set_ylabel(f"Wave {wave_id}", fontsize=20)

    for i, ax in enumerate(axes[0]):
        ax.set_title('{:.3f} '.format(time_steps[i].magnitude) + f'{time_steps[i].dimensionality}')

    for i in idx:
        x = waves_event.array_annotations['x_coords'][i]
        y = waves_event.array_annotations['y_coords'][i]
        t = waves_event.times[i]
        ax_i = np.where(time_steps == t)[0][0]

        channel = waves_event.array_annotations['channels'][i]
        t_i = np.argmax(asig.times >= t)
        i_start = np.max([0, t_i-half_window])
        i_stop = np.min([t_i+half_window, len(asig.times)-1])
        axes[0][ax_i].plot(asig.times[i_start : i_stop],
                           asig.as_array()[i_start : i_stop, channel])
        axes[0][ax_i].axvline(t, color='r')

        axes[1][ax_i].imshow(frames[t_i], origin='lower', cmap=cmap,
                             vmin=vmin, vmax=vmax)
        axes[1][ax_i].plot(x, y, linestyle='None', marker='D',
                           markersize=markersize, color='r')
        axes[1][ax_i].set_axis_off()

        axes[1][ax_i].quiver(x_idx[::skip_step,::skip_step],
                             y_idx[::skip_step,::skip_step],
                             np.real(vec_frames[t_i])[::skip_step,::skip_step],
                             np.imag(vec_frames[t_i])[::skip_step,::skip_step],
                             zorder=5)
    return axes


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    vec_asig = block.filter(name='optical_flow', objects="AnalogSignal")[0]

    frames = analogsignal_to_imagesequence(asig).as_array()
    vec_frames = analogsignal_to_imagesequence(vec_asig).as_array()

    waves_event = block.filter(name='wavefronts', objects="Event")[0]

    cmap = plt.get_cmap(args.colormap)

    for wave_id in np.unique(waves_event.labels):
        if int(wave_id) != -1:  # collection of not-clustered triggers
            ax = plot_wave(wave_id=wave_id,
                           waves_event=waves_event,
                           asig=asig,
                           frames=frames,
                           vec_frames=vec_frames,
                           time_window=args.time_window*pq.s,
                           cmap=cmap)

            output_path = os.path.join(args.output_dir,
                                       args.img_name.replace('id0', f'id{wave_id}'))
            save_plot(output_path)
            plt.close()
