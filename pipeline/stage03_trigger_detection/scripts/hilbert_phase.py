import neo
import numpy as np
import quantities as pq
from scipy.signal import hilbert
import argparse
import matplotlib.pyplot as plt


def detect_transitions(asig, transition_phase):
    # ToDo: replace with elephant function when signal can be neo object
    signal = asig.as_array()
    dim_t, channel_num = signal.shape

    hilbert_signal = hilbert(signal, axis=0)
    hilbert_phase = np.angle(hilbert_signal)

    # plt.plot(np.real(hilbert_signal[:250,5050]), color='r')
    # plt.plot(np.imag(hilbert_signal[:250,5050]), color='b')
    # plt.plot(hilbert_phase[:250,5050], color='g')
    # plt.show()

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
    channels = np.array([])

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
                             np.ones_like(channel_up_transitions)*channel_id)

    # save transitions as Event labels:'UP', array_annotations: channels
    sort_idx = np.argsort(up_transitions)

    return neo.Event(times=up_transitions[sort_idx]*asig.times.units,
                     labels=['UP'] * len(up_transitions),
                     name='Transitions',
                     array_annotations={'channels':channels[sort_idx]},
                     hilbert_transition_phase=transition_phase,
                     description='Transitions from down to up states. '\
                                +'annotated with the channel id ("channels").')


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--transition_phase", nargs='?', type=float)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    asig = block.segments[0].analogsignals[0]

    transition_event = detect_transitions(asig, args.transition_phase)

    block.segments[0].events.append(transition_event)

    with neo.NixIO(args.output) as io:
        io.write(block)
