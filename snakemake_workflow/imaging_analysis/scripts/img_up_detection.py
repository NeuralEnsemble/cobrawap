import neo
import numpy as np
import os
import argparse
import elephant as el
import itertools
import scipy as sc
import matplotlib.pyplot as plt


def filter_signals(images, lowcut, highcut, order):
    dim_t, dim_x, dim_y = images.shape
    coords = list(itertools.product(np.arange(dim_x), np.arange(dim_y)))
    images_array = images.as_array()
    for x,y in coords:
        if not np.isnan(np.sum(images_array[:,x,y])):
            images_array[:,x,y] = el.signal_processing.butter(images_array[:,x,y],
                                                        highpass_freq=lowcut,
                                                        lowpass_freq=highcut,
                                                        order=order,
                                                        fs=images.sampling_rate)
    return images_array


def detect_transitions(signal, times, transition_phase):
    # ToDo: replace with elephant function when signal can be neo object
    # hilbert_signal = el.signal_processing.hilbert(signal).as_array()
    if np.isnan(np.sum(signal)):
        return np.array([]) * times.units

    hilbert_signal = sc.signal.hilbert(signal)
    hilbert_phase = np.angle(hilbert_signal)

    peaks = []
    transitions = []
    crossings = [peaks, transitions]

    for i, phase in enumerate([0, transition_phase]):
        zero_crossings = np.where(np.diff(np.signbit(hilbert_phase-phase)))[0]
        for z in zero_crossings:
            if hilbert_phase[z] <= 0 \
            and np.real(hilbert_signal)[z] > np.imag(hilbert_signal)[z]:
                # positive crossing of phase
                crossings[i] += [times.magnitude[z]]

    up_transitions = np.array([])
    for i, peak in enumerate(peaks):
        dist = (peak - np.array(transitions))[peak - np.array(transitions) > 0]
        if len(dist):
            up_transitions = np.append(up_transitions,
                                       transitions[np.argmin(dist)])

    return up_transitions * times.units


def UP_detection(signals, times, t_start, t_stop, sampling_rate,
                 transition_phase, **annotations):
    dim_t, dim_x, dim_y = signals.shape
    up_trains = []
    for x in range(dim_x):
        for y in range(dim_y):
            ups = detect_transitions(signals[:,x,y], times, transition_phase)
            up_trains += [neo.core.SpikeTrain(ups,
                                              t_start=t_start,
                                              t_stop=t_stop,
                                              sampling_rate=sampling_rate,
                                              # minima_threshold=threshold,
                                              # minima_interplolation_window=window_size,
                                              coordinates=(x,y),
                                              grid_size=(dim_x,dim_y),
                                              transition_phase = transition_phase,
                                              transition_type='up',
                                              **annotations)]
    return up_trains


def remove_duplicate_properties(objects, del_keys=['nix_name', 'neo_name']):
    if type(objects) != list:
        objects = [objects]
    for i in range(len(objects)):
        for k in del_keys:
            if k in objects[i].annotations:
                del objects[i].annotations[k]
    return None


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--out_signal", nargs='?', type=str)
    CLI.add_argument("--transition_phase", nargs='?', type=float)
    # CLI.add_argument("--lowcut", nargs='?', type=float)
    # CLI.add_argument("--highcut", nargs='?', type=float)
    # CLI.add_argument("--order", nargs='?', type=int)
    # CLI.add_argument("--minima_threshold", nargs='?', type=float)
    # CLI.add_argument("--minima_windowsize", nargs='?', type=int)

    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        img_block = io.read_block()

    remove_duplicate_properties([img_block, img_block.segments[0]]
                                + img_block.segments[0].analogsignals)

    images = img_block.segments[0].analogsignals[0]

    # # Filter the signal
    # filt_signals = filter_signals(images, lowcut=args.lowcut,
    #                               highcut=args.highcut, order=args.order)

    up_trains = UP_detection(images.as_array(),
                             times=images.times,
                             t_start=images.t_start,
                             t_stop=images.t_stop,
                             # threshold=args.minima_threshold,
                             # window_size=args.minima_windowsize,
                             sampling_rate=images.sampling_rate,
                             transition_phase = args.transition_phase,
                             # filter_lowcut=args.lowcut,
                             # filter_highcut=args.highcut,
                             # filter_order=args.order,
                             **images.annotations)

    img_block.segments[0].spiketrains = up_trains
    with neo.NixIO(args.out_signal) as io:
        io.write(img_block)
