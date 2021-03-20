import argparse
import os
import numpy as np
import neo
import pandas as pd
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from utils import load_neo, AnalogSignal2ImageSequence


def label_planar(waves_event, vector_field, times, threshold):
    labels = np.unique(waves_event.labels)
    planarity = np.zeros(len(labels), dtype=float)

    for i, label in enumerate(labels):
        idx = np.where(label == waves_event.labels)[0]

        t_idx = [np.argmax(times >= t) for t
                 in waves_event.times[idx]]
        x = waves_event.array_annotations['x_coords'][idx]
        y = waves_event.array_annotations['y_coords'][idx]

        wave_directions = vector_field[t_idx, x.astype(int), y.astype(int)]
        norm = np.array([np.linalg.norm(w) for w in wave_directions])
        wave_directions /= norm

        planarity[i] = np.linalg.norm(np.mean(wave_directions))

    is_planar = planarity > threshold

    df =  pd.DataFrame(data=np.stack((planarity, is_planar), axis=1),
                       index=labels,
                       columns=['planarity', 'is_planar'])
    df['is_planar'] = df['is_planar'].astype(bool)
    df.index.name = 'wave_id'
    return df


def plot_planarity(waves_event, vector_field, times, wave_id, skip_step=1, ax=None):
    label = np.unique(waves_event.labels)[wave_id]

    idx = np.where(label == waves_event.labels)[0]

    t_idx = np.array([np.argmax(times >= t) for t
                      in waves_event.times[idx]])
    x = waves_event.array_annotations['x_coords'][idx]
    y = waves_event.array_annotations['y_coords'][idx]

    wave_directions = vector_field[t_idx, x.astype(int), y.astype(int)]
    norm = np.array([np.linalg.norm(w) for w in wave_directions])
    wave_directions /= norm

    palette = sns.husl_palette(len(np.unique(t_idx))+1, h=0.3, l=0.4)[:-1]

    if ax is None:
        fig, ax = plt.subplots()

    area = copy.copy(np.real(vector_field[0]))
    area[np.where(np.isfinite(area))] = 0
    ax.imshow(area, interpolation='nearest', origin='lower',
              vmin=-1, vmax=1, cmap='RdBu')

    for i, frame_t in enumerate(np.unique(t_idx)):
        frame_i = np.where(frame_t == t_idx)[0].astype(int)
        xi = x[frame_i]
        yi = y[frame_i]
        ti = t_idx[frame_i]
        frame = vector_field[ti, xi.astype(int), yi.astype(int)].magnitude
        ax.quiver(yi, xi, np.real(frame), np.imag(frame),
                  # units='width', scale=max(frame.shape)/(10*skip_step),
                  # width=0.15/max(frame.shape),
                  color=palette[i], alpha=0.8,
                  label='{:.3f} s'.format(times[frame_t].rescale('s').magnitude))

    dim_t, dim_x, dim_y = vector_field.as_array().shape
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(f'pixel size {vector_field.spatial_scale}')
    start_t = np.min(waves_event.times[idx]).rescale('s').magnitude
    stop_t = np.max(waves_event.times[idx]).rescale('s').magnitude
    ax.set_xlabel('{:.3f} - {:.3f} s'.format(start_t, stop_t))
    ax.set_title('planarity {:.3f}'.format(np.linalg.norm(np.mean(wave_directions))))
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    return ax


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--alignment_threshold", nargs='?', type=float, default=.85,
                     help="threshold for alignment of velocity vectors at transitions")

    args = CLI.parse_args()
    block = load_neo(args.data)

    block = AnalogSignal2ImageSequence(block)
    asig = block.segments[0].analogsignals[0]

    wavefront_evt = block.filter(name='Wavefronts', objects="Event")[0]

    optical_flow = block.filter(name='Optical Flow', objects="ImageSequence")[0]

    planar_labels = label_planar(waves_event=wavefront_evt,
                                 vector_field=optical_flow,
                                 times=asig.times,
                                 threshold=args.alignment_threshold)

    dim_t, dim_x, dim_y = optical_flow.shape
    skip_step = int(min([dim_x, dim_y]) / 50) + 1

    for i, wave_id in enumerate(np.unique(wavefront_evt.labels)):
        fig, ax = plt.subplots()
        plot_planarity(waves_event=wavefront_evt,
                       vector_field=optical_flow,
                       times=asig.times,
                       skip_step=skip_step,
                       wave_id=i,
                       ax=ax)
        plt.savefig(os.path.join(os.path.dirname(args.output), f'wave_{wave_id}.png'))
        plt.close()

    planar_labels.to_csv(args.output)
