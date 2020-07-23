import argparse
import os
import numpy as np
import neo
import pandas as pd
import matplotlib.pyplot as plt
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

    planar = planarity > threshold

    return pd.DataFrame(data=np.stack((labels, planarity, planar), axis=1),
                        index=labels,
                        columns=['wave_id', 'planarity', 'is_planar'])


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

    if ax is None:
        fig, ax = plt.subplots()

    palette = sns.color_palette('Set2', len(np.unique(t_idx)))

    for i, frame_t in enumerate(np.unique(t_idx)):
        frame_i = np.where(frame_t == t_idx)[0].astype(int)
        xi = x[frame_i]
        yi = y[frame_i]
        ti = t_idx[frame_i]
        frame = vector_field[ti, xi.astype(int), yi.astype(int)].magnitude
        ax.quiver(xi, yi, -np.real(frame), -np.imag(frame),
                  color=palette[i], alpha=0.5, label=f'{asig.times[frame_t]}')

    dim_t, dim_x, dim_y = vector_field.as_array().shape
    ax.set_xlim((0, dim_x))
    ax.set_ylim((0, dim_y))
    ax.set_xlabel(f'x scale {vector_field.spatial_scale}')
    ax.set_ylabel(f'y scale {vector_field.spatial_scale}')
    ax.set_title('planarity {:.3f}'.format(np.linalg.norm(np.mean(wave_directions))))
    plt.legend()
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

    wavefront_evt = [evt for evt in block.segments[0].events
                     if evt.name == "Wavefronts"]
    if wavefront_evt:
        wavefront_evt = wavefront_evt[0]
    else:
        raise ValueError("Input does not contain an event with name " \
                       + "'Wavefronts'!")

    optical_flow = [imgseq for imgseq in block.segments[0].imagesequences
                    if imgseq.name == "Optical Flow"]

    if optical_flow:
        optical_flow = optical_flow[0]
    else:
        raise ValueError("Input does not contain an event with name " \
                       + "'Optical Flow'!")

    planar_labels = label_planar(waves_event=wavefront_evt,
                                 vector_field=optical_flow,
                                 times = asig.times,
                                 threshold=args.alignment_threshold)

    for i, wave_id in enumerate(np.unique(wavefront_evt.labels)):
        fig, ax = plt.subplots()
        plot_planarity(waves_event=wavefront_evt,
                       vector_field=optical_flow,
                       times=asig.times,
                       wave_id=i,
                       ax=ax)
        plt.savefig(os.path.join(os.path.dirname(args.output), f'wave_{wave_id}.png'))
        plt.close()

    planar_labels.to_csv(args.output)
