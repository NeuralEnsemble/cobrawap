import neo
import numpy as np
import os
import argparse
import elephant as el
import itertools
import scipy as sc
from copy import copy, deepcopy
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


def detect_minima(signal, times, threshold, window_size):
    argmins = sc.signal.argrelmin(signal, order=2)[0]
    argmins = np.array([argmin for argmin in argmins if argmin > threshold])
    interpolated_mins = np.array([])
    for argmin in argmins:
        i_start = max(0, argmin - int(window_size/2))
        i_stop = min(len(signal)-1, argmin + int(window_size/2)) + 1
        params = np.polyfit(times[i_start:i_stop],
                            signal[i_start:i_stop],
                            deg=2)
        interpolated_min = -params[1]/(2*params[0])
        if min(times) < interpolated_min < max(times):
            interpolated_mins = np.append(interpolated_mins, interpolated_min)
    return interpolated_mins * times.units


def UP_detection(signals, times, threshold, window_size,
                 t_start, t_stop, sampling_rate, **annotations):
    dim_t, dim_x, dim_y = signals.shape
    up_trains = []
    for x in range(dim_x):
        for y in range(dim_y):
            ups = detect_minima(signals[:,x,y], times, threshold, window_size)
            up_trains += [neo.core.SpikeTrain(ups,
                                              t_start=t_start,
                                              t_stop=t_stop,
                                              sampling_rate=sampling_rate,
                                              minima_threshold=threshold,
                                              minima_interplolation_window=window_size,
                                              position=(x,y),
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
    CLI.add_argument("--out_image", nargs='?', type=str)
    CLI.add_argument("--out_signal", nargs='?', type=str)
    CLI.add_argument("--lowcut", nargs='?', type=float)
    CLI.add_argument("--highcut", nargs='?', type=float)
    CLI.add_argument("--order", nargs='?', type=int)
    CLI.add_argument("--minima_threshold", nargs='?', type=float)
    CLI.add_argument("--minima_windowsize", nargs='?', type=int)

    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        img_block = io.read_block()

    remove_duplicate_properties([img_block, img_block.segments[0]]
                                + img_block.segments[0].analogsignals)

    images = img_block.segments[0].analogsignals[0]

    example_pixel = (int(images.shape[1]/2), int(images.shape[2]/2))
    example_image = copy(images.as_array()[:,example_pixel[0],example_pixel[1]])

    # Filter the signal
    filt_signals = filter_signals(images, lowcut=args.lowcut,
                                  highcut=args.highcut, order=args.order)

    up_trains = UP_detection(filt_signals,
                             times=images.times,
                             t_start=images.t_start,
                             t_stop=images.t_stop,
                             threshold=args.minima_threshold,
                             window_size=args.minima_windowsize,
                             sampling_rate=images.sampling_rate,
                             filter_lowcut=args.lowcut,
                             filter_highcut=args.highcut,
                             filter_order=args.order,
                             **images.annotations)

    img_block.segments[0].spiketrains = up_trains
    with neo.NixIO(args.out_signal) as io:
        io.write(img_block)


    print(np.where([st.annotations['position'] for st in up_trains] == example_pixel))
    fig, ax = plt.subplots()
    ax.plot(images.times, example_image, c='b', label='clean signal')
    ax.plot(images.times, filt_signals[:,example_pixel[0],example_pixel[1]],
            c='r', label='filtered [{} - {}] Hz'.format(args.lowcut, args.highcut))
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'Ca$^+$ signal')
    ax.set_title('signal of pixel {}'.format(example_pixel))
    plt.legend()
    plt.savefig(args.out_image)
    plt.show()
