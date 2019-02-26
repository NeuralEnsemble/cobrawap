from load_and_transform_to_neo import load_segment
import argparse
import matplotlib.pyplot as plt
import elephant as el
import numpy as np


def plot_power_spectrum(segment, psd_num_seg, psd_overlap):
    fig, ax = plt.subplots()
    handles = {}
    fs = segment.analogsignals[0].sampling_rate.rescale('1/s').magnitude

    for asig_count, asig in enumerate(segment.analogsignals):

        (f, p) = el.spectral.welch_psd(np.squeeze(asig),
                                       num_seg=psd_num_seg, overlap=psd_overlap,
                                       window='hanning', nfft=None, fs=fs,
                                       detrend='constant', return_onesided=True,
                                       scaling='density', axis=-1)

        handle, = ax.semilogy(f, p, alpha=0.7, color=asig.annotations['electrode_color'])
        handles[asig.annotations['cortical_location']] = handle
    # ax.set_xlim((0, 1500))
    # ax.set_ylim((10**(-2), 10**0))
    ax.set_ylabel('spectral density')
    ax.set_xlabel('frequency [Hz]')
    plt.legend([handle for handle in handles.values()],
               [location for location in handles.keys()], loc=1)
    return None


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs=1, type=str)
    CLI.add_argument("--data",      nargs=1, type=str)
    CLI.add_argument("--format",    nargs=1, type=str, default='eps')
    CLI.add_argument("--show_figure",   nargs=1, type=int, default=0)
    CLI.add_argument("--psd_num_seg",  nargs=1, type=int)
    CLI.add_argument("--psd_overlap",  nargs=1, type=float)

    args = CLI.parse_args()

    segment = load_segment(filename=args.data[0])

    plot_power_spectrum(segment,
                        psd_num_seg=args.psd_num_seg[0],
                        psd_overlap=args.overlap[0])

    if args.show_figure[0]:
        plt.show()

    plt.savefig(fname=args.output[0], format=args.format[0])
