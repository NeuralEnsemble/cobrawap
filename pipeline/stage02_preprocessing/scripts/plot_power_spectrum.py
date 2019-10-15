import neo
import numpy as np
import argparse
import matplotlib.pyplot as plt
from elephant.spectral import welch_psd
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape



def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data", nargs='?', type=str)
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--highpass_freq", nargs='?', type=none_or_float)
    CLI.add_argument("--lowpass_freq", nargs='?', type=none_or_float)
    CLI.add_argument("--psd_freq_res", nargs='?', type=float)
    CLI.add_argument("--psd_overlap", nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)

    freqs, psd = welch_psd(block.segments[0].analogsignals[0],
                           freq_res=args.psd_freq_res*pq.Hz,
                           overlap=args.psd_overlap)

    fig, ax = plt.subplots()
    # ToDo: plot also the channel-wise power spectra ?
    ax.plot(freqs, np.mean(psd, axis=0))
    ax.set_title('Power Spectrum')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('average power density')

    if args.highpass_freq is not None and args.lowpass_freq is not None:
        left = args.highpass_freq if args.highpass_freq is not None else 0
        right = args.lowpass_freq if args.lowpass_freq is not None else ax.get_xlim()[1]
        ax.axvspan(left, right, alpha=0.3, color='k', label='filtered region')

    plt.savefig(args.output, bbox_inches="tight")
