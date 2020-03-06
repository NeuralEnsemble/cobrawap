import argparse
import os
import matplotlib.pyplot as plt
import elephant as el
import numpy as np
import neo


def plot_power_spectrum(segment, psd_freq_res, psd_overlap):
    fig, ax = plt.subplots()
    handles = {}
    for asig_count, asig in enumerate(segment.analogsignals):
        (f, p) = el.spectral.welch_psd(asig,
                                       freq_res=psd_freq_res, overlap=psd_overlap,
                                       window='hanning', nfft=None,
                                       detrend='constant', return_onesided=True,
                                       scaling='density', axis=-1)
        handle, = ax.semilogy(f, np.squeeze(p), alpha=0.7, color=asig.annotations['electrode_color'])
        handles[asig.annotations['cortical_location']] = handle
    ax.set_ylabel('spectral density')
    ax.set_xlabel('frequency [Hz]')
    plt.legend([handle for handle in handles.values()],
               [location for location in handles.keys()], loc=1)
    return None


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--format",    nargs='?', type=str, default='eps')
    CLI.add_argument("--show_figure",   nargs='?', type=int, default=0)
    CLI.add_argument("--psd_freq_res",  nargs='?', type=int)
    CLI.add_argument("--psd_overlap",  nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        segment = io.read_block().segments[0]

    plot_power_spectrum(segment,
                        psd_freq_res=args.psd_freq_res,
                        psd_overlap=args.psd_overlap)

    if args.show_figure:
        plt.show()

    data_dir = os.path.dirname(args.output)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(fname=args.output, format=args.format)
