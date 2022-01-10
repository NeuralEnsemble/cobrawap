import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from elephant.spectral import welch_psd
from utils.io import load_neo, save_plot
from utils.parse import none_or_float


def plot_psd(freqs, psd, highpass_freq, lowpass_freq):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()
    for channel in psd:
        ax.semilogy(freqs, channel, alpha=0.7)
    ax.semilogy(freqs, np.mean(psd, axis=0), linewidth=2,
                color='k', label='channel average')
    ax.set_title('Power Spectrum')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectral density')

    if highpass_freq is not None and lowpass_freq is not None:
        left = highpass_freq if highpass_freq is not None else 0
        right = lowpass_freq if lowpass_freq is not None else ax.get_xlim()[1]
        ax.axvspan(left, right, alpha=0.2, color='k', label='filtered region')

    plt.legend()
    return fig


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output figure")
    CLI.add_argument("--highpass_freq", nargs='?', type=none_or_float,
                     default='None', help="lower bound of frequency band in Hz")
    CLI.add_argument("--lowpass_freq", nargs='?', type=none_or_float,
                     default='None', help="upper bound of frequency band in Hz")
    CLI.add_argument("--psd_freq_res", nargs='?', type=float, default=5,
                     help="frequency resolution of the power spectrum in Hz")
    CLI.add_argument("--psd_overlap", nargs='?', type=float, default=0.5,
                     help="overlap parameter for Welch's algorithm [0-1]")
    args = CLI.parse_args()

    asig = load_neo(args.data, 'analogsignal')

    freqs, psd = welch_psd(asig,
                           freq_res=args.psd_freq_res*pq.Hz,
                           overlap=args.psd_overlap)

    plot_psd(freqs=freqs,
             psd=psd,
             highpass_freq=args.highpass_freq,
             lowpass_freq=args.lowpass_freq)

    save_plot(args.output)
