"""
Create a plot of the channel-wise and average power spectrum density.
"""

import numpy as np
import quantities as pq
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from elephant.spectral import welch_psd
from utils.io_utils import load_neo, save_plot
from utils.parse import none_or_float

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output figure")
CLI.add_argument("--highpass_frequency", nargs='?', type=none_or_float,
                 default='None', help="lower bound of frequency band in Hz")
CLI.add_argument("--lowpass_frequency", nargs='?', type=none_or_float,
                 default='None', help="upper bound of frequency band in Hz")
CLI.add_argument("--psd_frequency_resolution", nargs='?', type=float, default=5,
                 help="frequency resolution of the power spectrum in Hz")
CLI.add_argument("--psd_overlap", nargs='?', type=float, default=0.5,
                 help="overlap parameter for Welch's algorithm [0-1]")

def plot_psd(frequencies, psd, highpass_frequency, lowpass_frequency):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()
    for channel in psd:
        ax.semilogy(frequencies, channel, alpha=0.7)
    ax.semilogy(frequencies, np.mean(psd, axis=0), linewidth=2,
                color='k', label='channel average')
    ax.set_title('Power Spectrum')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectral density')

    if highpass_frequency is not None and lowpass_frequency is not None:
        left = highpass_frequency if highpass_frequency is not None else 0
        right = lowpass_frequency if lowpass_frequency is not None else ax.get_xlim()[1]
        ax.axvspan(left, right, alpha=0.2, color='k', label='filtered region')

    plt.legend()
    return fig


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    asig = load_neo(args.data, 'analogsignal')

    freqs, psd = welch_psd(asig,
                           frequency_resolution=args.psd_frequency_resolution*pq.Hz,
                           window='hann',
                           overlap=args.psd_overlap)

    plot_psd(frequencies=freqs,
             psd=psd,
             highpass_frequency=args.highpass_frequency,
             lowpass_frequency=args.lowpass_frequency)

    save_plot(args.output)
