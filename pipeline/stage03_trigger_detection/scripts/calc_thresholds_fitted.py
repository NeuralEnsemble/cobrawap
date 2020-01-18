import neo
import numpy as np
import random
import argparse
import os
import scipy as sc
import warnings
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt

warnings.simplefilter("error", OptimizeWarning)


def fit_amplitude_distribution(signal, sigma_factor, fit_function,
                               bins, plot_channel):
    # signal amplitude distribution
    signal = signal[np.isfinite(signal)]
    hist, edges = np.histogram(signal, bins=bins, density=True)
    xvalues = edges[:-1] + np.diff(edges)[0] / 2.

    if fit_function == 'Gaussian':
        fit_func = lambda x, m, s: 1. / (s * np.sqrt(2 * np.pi))\
                                 * np.exp(-0.5 * ((x - m) / s) ** 2)
    else:
        raise NotImplementedError('The fitting function {} is not yet implementd!'\
                                  .format(fit_function))

    # First fit -> determine peak location m0
    try:
        (m0, _), _ = sc.optimize.curve_fit(fit_func, xvalues, hist, p0=(0, 1))
    except OptimizeWarning:
        print('Could not perform first fit. Using Median to determine central downstate signal amplitude.')
        m0 = np.median(signal)

    # shifting to 0
    xvalues -= m0

    # Mirror left peak side for 2nd fit
    signal_leftpeak = signal[signal - m0 <= 0] - m0
    mirror_signal = np.append(signal_leftpeak, -1 * signal_leftpeak)
    peakhist, edges = np.histogram(mirror_signal, bins=bins, density=True)
    xvalues2 = edges[:-1] + np.diff(edges)[0] / 2.

    # Second fit -> determine spread s0
    try:
        (_, s0), _ = sc.optimize.curve_fit(fit_func, xvalues2, peakhist, p0=(0, np.std(peakhist)))
    except OptimizeWarning:
        print('Could not perform second fit. Using std to determine spread of downstate signal amplitudes.')
        s0 = np.std(peakhist)

    ## PLOTTING ##
    if plot_channel:
        fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
        ax[0].bar(xvalues, hist, width=np.diff(xvalues)[0], color='r')
        left_right_ratio = len(signal_leftpeak) * 2. / len(signal)
        ax[0].plot(xvalues, [left_right_ratio * fit_func(x, 0, s0) for x in xvalues], c='k')
        ax[0].set_xlabel('signal')
        ax[0].set_ylabel('sample density')
        ax[0].set_title('Amplitude distribution (channel {})'.format(plot_channel))

        ax[1].bar(xvalues, [hist[i] - fit_func(x, 0, s0) for (i, x) in enumerate(xvalues)],
                  width=np.diff(xvalues)[0], color='r')
        ax[1].set_xlabel('signal')
        ax[1].set_title('tail')
        ax[1].axvline(sigma_factor * s0, color='k', ls='--'),
        ax[1].text(1.1 * sigma_factor * s0, 0.9 * ax[1].get_ylim()[0],
                   r'UD threshold ({}$\sigma$)'.format(sigma_factor), color='k')

    return m0 + sigma_factor * s0

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--output_img",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--fit_function", nargs='?', type=str)
    CLI.add_argument("--sigma_factor", nargs='?', type=float)
    CLI.add_argument("--bin_num", nargs='?', type=int)
    CLI.add_argument("--plot_channel", nargs='?', type=none_or_int)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        asig = io.read_block().segments[0].analogsignals[0]

    signal = asig.as_array()
    dim_t, dim_channels = signal.shape

    if args.plot_channel is None:
        args.plot_channel = random.randint(0, dim_channels)

    thresholds = np.zeros(dim_channels)

    for channel in np.arange(dim_channels):
        if channel == args.plot_channel:
            plot_channel = channel
        else:
            plot_channel = False
        thresholds[channel] = fit_amplitude_distribution(signal[:,channel],
                                                         args.sigma_factor,
                                                         args.fit_function,
                                                         args.bin_num,
                                                         plot_channel)
        if plot_channel:
            plt.savefig(args.output_img)


    np.save(args.output, thresholds)
