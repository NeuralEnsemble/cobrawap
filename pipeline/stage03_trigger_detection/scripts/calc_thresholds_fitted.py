import neo
import numpy as np
import random
import argparse
import os
import scipy as sc
import warnings
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
from utils import load_neo, save_plot, none_or_int

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


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output thresholds (numpy array)")
    CLI.add_argument("--output_img", nargs='?', type=lambda v: v.split(','),
                     default=None, help="path(s) of output figure(s)")
    CLI.add_argument("--fit_function", nargs='?', type=str, default='Gaussian',
                     help="function to fit the amplitude distribution")
    CLI.add_argument("--sigma_factor", nargs='?', type=float, default=3,
                     help="sigma_factor x standard deviation = threshold")
    CLI.add_argument("--bin_num", nargs='?', type=int, default=100,
                     help='number of bins for the amplitude histogram')
    CLI.add_argument("--plot_channels", nargs='+', type=none_or_int,
                     default=None, help="list of channels to plot")
    args = CLI.parse_args()

    asig = load_neo(args.data, 'analogsignal')

    signal = asig.as_array()
    dim_t, dim_channels = signal.shape

    thresholds = np.zeros(dim_channels)

    for channel in np.arange(dim_channels):
        if channel in args.plot_channels:
            plot_channel = channel
        else:
            plot_channel = False
        thresholds[channel] = fit_amplitude_distribution(signal[:,channel],
                                                         args.sigma_factor,
                                                         args.fit_function,
                                                         args.bin_num,
                                                         plot_channel)
        if plot_channel:
            fig_idx = np.where(channel == args.plot_channels)[0]
            save_plot(args.output_img[fig_idx])

    np.save(args.output, thresholds)
