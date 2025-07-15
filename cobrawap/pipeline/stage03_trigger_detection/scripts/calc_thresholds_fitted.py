"""
Determine the threshold between Up and Down states for each channel
by fitting the respective amplitude distributions.
"""

import numpy as np
import argparse
from pathlib import Path
import os
import scipy as sc
import warnings
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_int

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output thresholds (numpy array)")
CLI.add_argument("--img_dir", nargs='?', type=Path, default=None,
                 help="path of output figure directory")
CLI.add_argument("--img_name", nargs='?', type=str, default=None,
                 help="example name of output figure for channel 0")
CLI.add_argument("--fit_function", nargs='?', type=str, default='Gaussian',
                 help="function to fit the amplitude distribution")
CLI.add_argument("--sigma_factor", nargs='?', type=float, default=3,
                 help="sigma_factor x standard deviation = threshold")
CLI.add_argument("--bin_num", nargs='?', type=int, default=100,
                 help='number of bins for the amplitude histogram')
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int,
                 default=None, help="list of channels to plot")

warnings.simplefilter("error", OptimizeWarning)

def gaussian(x, mu=0, sig=1):
    return 1. / (sig * np.sqrt(2 * np.pi)) \
         * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def double_gaussian(x, ratio=.5, mu1=0, sig1=1, mu2=0, sig2=1):
    return ratio * gaussian(x=x, mu=mu1, sig=sig1) \
         + (1-ratio) * gaussian(x=x, mu=mu2, sig=sig2)

def double_gaussian_fit(params, x, y):
    ratio, mu1, sig1, mu2, sig2 = params
    fit = double_gaussian(x, ratio=ratio, mu1=mu1, sig1=sig1, mu2=mu2, sig2=sig2)
    return fit - y


def fit_amplitude_distribution(signal, sigma_factor, fit_function,
                               bins, plot_channel):
    # signal amplitude distribution
    signal = signal[np.isfinite(signal)]
    hist, edges = np.histogram(signal, bins=bins, density=True)
    xvalues = edges[:-1] + np.diff(edges)[0] / 2.

    if fit_function == 'HalfGaussian':
        # First fit -> determine peak location m0
        try:
            (m0, _), _ = sc.optimize.curve_fit(gaussian, xvalues, hist, p0=(0, 1))
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
            (_, s0), _ = sc.optimize.curve_fit(fit_function, xvalues2, peakhist, p0=(0, np.std(peakhist)))
        except OptimizeWarning:
            print('Could not perform second fit. Using std to determine spread of downstate signal amplitudes.')
            s0 = np.std(peakhist)
        except RuntimeError:
            print('Could not perform second fit. Using std to determine spread of downstate signal amplitudes.')
            s0 = np.std(peakhist)

        if plot_channel:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.bar(xvalues, hist, width=np.diff(xvalues)[0], color='r')
            left_right_ratio = len(signal_leftpeak) * 2. / len(signal)
            ax.plot(xvalues, [left_right_ratio * fit_function(x, 0, s0) for x in xvalues], c='k')
            ax.set_xlabel('signal')
            ax.set_ylabel('sample density')
            ax.set_title('Amplitude distribution (channel {})'.format(plot_channel))
            ax.axvline(sigma_factor * s0, color='k', ls='--'),
            ax.text(1.1 * sigma_factor * s0, 0.9 * ax[1].get_ylim()[0],
                       r'UD threshold ({}$\sigma$)'.format(sigma_factor), color='k')
        return m0 + sigma_factor * s0

    elif fit_function == 'DoubleGaussian':
        # ratio, mu1, sig1, mu2, sig2
        inital_params = np.array([.5, xvalues[0], .1, xvalues[-1], .1])
        mu_bound = (xvalues[0], xvalues[-1])
        sig_bound = (0, xvalues[-1]-xvalues[0])
        bounds = ([0, mu_bound[0], sig_bound[0], mu_bound[0], sig_bound[0]],
                  [1, mu_bound[1], sig_bound[1], mu_bound[1], sig_bound[1]])

        optimize_result = sc.optimize.least_squares(double_gaussian_fit,
                                                    inital_params,
                                                    bounds=bounds,
                                                    kwargs={'x':xvalues,
                                                            'y':hist})
        ratio, mu1, sig1, mu2, sig2 = optimize_result.x

        x = np.linspace(mu1, mu2, 100)
        min_idx = sc.signal.argrelmin(double_gaussian(x, ratio=ratio, mu1=mu1,
                                                      sig1=sig1, mu2=mu2, sig2=sig2))
        if len(min_idx[0]):
            threshold = x[min_idx[0][0]]
        else:
            threshold = mu1 + sigma_factor*sig1

        if plot_channel:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.bar(xvalues, hist, width=np.diff(xvalues)[0], color='r')
            ax.plot(xvalues, double_gaussian(xvalues, ratio=ratio, mu1=mu1, sig1=sig1, mu2=mu2, sig2=sig2))
            ax.axvline(threshold, linestyle=':', color='k')
        return threshold
    else:
        raise NotImplementedError('The fitting function {} is not yet implementd!'\
                                  .format(fit_function))

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()
    
    asig = load_neo(args.data, 'analogsignal')
    signal = asig.as_array()
    dim_t, dim_channels = signal.shape
    non_nan_channels = [i for i in range(dim_channels) if np.isfinite(signal[:,i]).all()]

    thresholds = np.empty(dim_channels)
    thresholds.fill(np.nan)

    for channel in non_nan_channels:
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
            output_path = os.path.join(args.img_dir,
                                       args.img_name.replace('_channel0', f'_channel{channel}'))
            save_plot(output_path)

    np.save(args.output, thresholds)
