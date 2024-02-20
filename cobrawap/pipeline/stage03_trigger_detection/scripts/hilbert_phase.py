"""
Detect trigger times (i.e., state transition / local wavefronts onsets) 
by finding crossing of a set phase-value in the channel signals.
"""

import argparse
import neo
import numpy as np
import quantities as pq
from scipy.signal import hilbert
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io_utils import load_neo, write_neo, save_plot
from utils.neo_utils import time_slice, remove_annotations
from utils.parse import none_or_int, none_or_float
from pathlib import Path

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--img_dir", nargs='?', type=Path,
                 default=None, help="path of figure directory")
CLI.add_argument("--img_name", nargs='?', type=str,
                 default='hilbert_phase_channel0.png',
                 help='example image filename for channel 0')
CLI.add_argument("--transition_phase", nargs='?', type=float, default=-1.570796,
                 help="phase to use as threshold for the upward transition")
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                 help="list of channels to plot")
CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0,
                 help="start time in seconds")
CLI.add_argument("--plot_tstop",  nargs='?', type=none_or_float, default=10,
                 help="stop time in seconds")

def detect_transitions(asig, transition_phase):
    # ToDo: replace with elephant function
    signal = asig.as_array()
    dim_t, channel_num = signal.shape

    hilbert_signal = hilbert(signal, axis=0)
    hilbert_phase = np.angle(hilbert_signal)

    def _detect_phase_crossings(phase):
        # detect phase crossings from below phase to above phase
        is_larger = hilbert_phase > phase
        positive_crossings = ~is_larger & np.roll(is_larger, -1, axis=0)
        positive_crossings = positive_crossings[:-1]

        # select phases within [-pi, pi]
        real_crossings = np.real(hilbert_signal[:-1]) > np.imag(hilbert_signal[:-1])
        crossings = real_crossings & positive_crossings

        # arrange transitions times per channel
        times = asig.times[:-1]
        crossings_list = [times[crossings[:,channel]].magnitude
                          for channel in range(channel_num)]
        return crossings_list

    # UP transitions: A change of the hilbert phase from < transtion_phase
    #                 to > transition_phase, followed by a peak (phase = 0).

    peaks = _detect_phase_crossings(0)
    transitions = _detect_phase_crossings(transition_phase)

    up_transitions = np.array([])
    channels = np.array([], dtype=int)

    for channel_id, (channel_peaks, channel_transitions) in enumerate(zip(peaks, transitions)):
        channel_up_transitions = np.array([])
        if channel_peaks is not None:
            for peak in channel_peaks:
                distance_to_peak = peak - np.array(channel_transitions)
                distance_to_peak = distance_to_peak[distance_to_peak > 0]
                if distance_to_peak.size:
                    trans_idx = np.argmin(distance_to_peak)
                    channel_up_transitions = np.append(channel_up_transitions,
                                                       channel_transitions[trans_idx])
        channel_up_transitions = np.unique(channel_up_transitions)
        up_transitions = np.append(up_transitions, channel_up_transitions)
        channels = np.append(channels,
                             np.ones_like(channel_up_transitions, dtype=int)*channel_id)

    # save transitions as Event labels:'UP', array_annotations: channels
    sort_idx = np.argsort(up_transitions)

    evt = neo.Event(times=up_transitions[sort_idx]*asig.times.units,
                    labels=['UP'] * len(up_transitions),
                    name='transitions',
                    array_annotations={'channels':channels[sort_idx]},
                    trigger_detection='hilbert_phase',
                    hilbert_transition_phase=transition_phase,
                    description='transitions from down to up states. '\
                               +'annotated with the channel id ("channels").')

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channels[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)
    return evt


def plot_hilbert_phase(asig, event, channel):
    signal = asig.as_array()[:,channel]

    hilbert_signal = hilbert(signal, axis=0)
    hilbert_phase = np.angle(hilbert_signal)

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    ax.plot(asig.times.rescale('s'), zscore(signal), label='signal')
    ax.plot(asig.times.rescale('s'), hilbert_phase, label='hilbert phase')

    for t, c in zip(event.times, event.array_annotations['channels']):
        if c == channel:
            ax.axvline(t.rescale('s'), color='k')

    ax.set_title('Channel {}'.format(channel))
    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))
    ax.legend()
    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]

    args.plot_tstart = asig.t_start if args.plot_tstart is None else args.plot_tstart
    args.plot_tstop = asig.t_stop if args.plot_tstop is None else args.plot_tstop

    transition_event = detect_transitions(asig, args.transition_phase)

    block.segments[0].events.append(transition_event)

    write_neo(args.output, block)

    if args.plot_channels[0] is not None:
        for channel in args.plot_channels:
            plot_hilbert_phase(asig=time_slice(asig, args.plot_tstart, args.plot_tstop),
                               event=time_slice(transition_event, args.plot_tstart, args.plot_tstop),
                               channel=int(channel))
            output_path = args.img_dir / args.img_name.replace('_channel0', f'_channel{channel}')
            save_plot(output_path)
