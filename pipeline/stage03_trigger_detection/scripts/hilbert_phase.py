import neo
import numpy as np
import quantities as pq
from scipy.signal import hilbert
from scipy.stats import zscore
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_neo, write_neo, time_slice, none_or_int,\
                  remove_annotations, save_plot

def detect_transitions(asig, transition_phase):
    # ToDo: replace with elephant function
    signal = asig.as_array()
    dim_t, channel_num = signal.shape

    hilbert_signal = hilbert(signal, axis=0)
    hilbert_phase = np.angle(hilbert_signal)

    def _detect_phase_crossings(phase):
        t_idx, channel_idx = np.where(np.diff(np.signbit(hilbert_phase-phase), axis=0))
        crossings = [None] * channel_num
        for ti, channel in zip(t_idx, channel_idx):
            # select only crossings from negative to positive
            if (hilbert_phase-phase)[ti][channel] <= 0 \
            and np.real(hilbert_signal[ti][channel]) \
              > np.imag(hilbert_signal[ti][channel]):
                if crossings[channel] is None:
                    crossings[channel] = np.array([])
                if asig.times[ti].magnitude not in crossings[channel]:
                    crossings[channel] = np.append(crossings[channel],
                                                   asig.times[ti].magnitude)
        return crossings

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
                     name='Transitions',
                     array_annotations={'channels':channels[sort_idx]},
                     hilbert_transition_phase=transition_phase,
                     description='Transitions from down to up states. '\
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
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=lambda v: v.split(','),
                     default='None', help="path(s) of output figure(s)")
    CLI.add_argument("--transition_phase", nargs='?', type=float, default=-1.570796,
                     help="phase to use as threshold for the upward transition")
    CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                     help="list of channels to plot")
    CLI.add_argument("--plot_tstart", nargs='?', type=float, default=0,
                     help="start time in seconds")
    CLI.add_argument("--plot_tstop",  nargs='?', type=float, default=10,
                     help="stop time in seconds")
    args = CLI.parse_args()

    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]

    transition_event = detect_transitions(asig, args.transition_phase)

    block.segments[0].events.append(transition_event)

    write_neo(args.output, block)

    if args.plot_channels[0] is not None:
        if not len(args.output_img) == len(args.plot_channels):
            raise InputError("The number of plotting channels must "\
                           + "correspond to the number of image output paths!")

        for output, channel in zip(args.output_img, args.plot_channels):
            plot_hilbert_phase(asig=time_slice(asig, args.plot_tstart, args.plot_tstop),
                               event=time_slice(transition_event, args.plot_tstart, args.plot_tstop),
                               channel=int(channel))
            save_plot(output)
