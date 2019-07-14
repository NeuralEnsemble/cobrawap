import neo
import numpy as np
import os
import argparse
import elephant as el
import itertools
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


if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--out_image", nargs='?', type=str)
    CLI.add_argument("--out_signal", nargs='?', type=str)
    CLI.add_argument("--lowcut", nargs='?', type=float)
    CLI.add_argument("--highcut", nargs='?', type=float)
    CLI.add_argument("--order", nargs='?', type=int)
    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        images = io.read_block().segments[0].analogsignals[0]

    fig, ax = plt.subplots()
    ax.plot(images.times, images.as_array()[:,30,30], c='b',
            label='clean signal')


    # Filter the signal
    filt_signals = filter_signals(images, lowcut=args.lowcut,
                                  highcut=args.highcut, order=args.order)

    ax.plot(images.times, filt_signals[:,30,30], c='r',
             label='filtered [0.5 - 3.0] Hz')
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'Ca$^+$ signal')
    ax.set_title('signal of pixel (30,30)')
    plt.legend()
    plt.show()
