"""
Estimate the logarithmic multi-unit activity (MUA) by averaging the power
in a given frequency range.
"""

import numpy as np
from elephant.spectral import welch_psd
from elephant.signal_processing import butter
from scipy.stats import zscore
import quantities as pq
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
from utils.io_utils import load_neo, write_neo, save_plot
from utils.parse import none_or_float, none_or_int
from utils.neo_utils import time_slice

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--img_dir", nargs='?', type=Path,
                 default=None, help="path of figure directory")
CLI.add_argument("--img_name", nargs='?', type=str,
                 default='logMUA_trace_channel0.png',
                 help='example image filename for channel 0')
CLI.add_argument("--highpass_frequency", nargs='?', type=float, default=200,
                 help="lower bound of frequency band in Hz")
CLI.add_argument("--lowpass_frequency", nargs='?', type=float, default=1500,
                 help="upper bound of frequency band in Hz")
CLI.add_argument("--logMUA_rate", nargs='?', type=none_or_float, default=None,
                 help="rate of the signal after transformation")
CLI.add_argument("--psd_overlap", nargs='?', type=float, default=0.5,
                 help="overlap parameter for Welch's algorithm [0-1]")
CLI.add_argument("--fft_slice", nargs='?', type=none_or_float, default=None,
                 help="time window length used for power spectrum estimate, in s")
CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0,
                 help="start time in seconds")
CLI.add_argument("--plot_tstop", nargs='?', type=none_or_float, default=10,
                 help="stop time in seconds")
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=None,
                 help="list of channels to plot")


def logMUA_estimation(asig, highpass_frequency, lowpass_frequency, logMUA_rate,
                      psd_overlap, fft_slice):
    time_steps, channel_num = asig.shape
    fs = asig.sampling_rate.rescale('Hz')
    if fft_slice is None:
        fft_slice = (1/highpass_frequency).rescale('s')
    elif fft_slice < 1/highpass_frequency:
        raise ValueError("Too few fft samples to estimate the frequency "\
                       + "content in the range [{} {}]."\
                         . format(highpass_frequency, lowpass_frequency))
    # logMUA_rate can only be an int fraction of the orginal sampling_rate
    if logMUA_rate is None:
        logMUA_rate = highpass_frequency
    if logMUA_rate > fs:
        raise ValueError("The requested logMUA rate can not be larger than "\
                       + "the inital sampling rate!")
    subsample_order = int(fs/logMUA_rate)
    eff_logMUA_rate = fs/subsample_order
    print("effective logMUA rate = {}".format(eff_logMUA_rate))

    if 1/eff_logMUA_rate > fft_slice:
        raise ValueError("The given logMUA_rate is too low to capture "\
                       + "the logMUA estimate of all the signal with "\
                       + "sample size {}. ".format(fft_slice)\
                       + "Either increase the logMUA_rate "\
                       + "or increase fft_slice.")

    subsample_times = np.arange(asig.t_start.rescale('s'),
                                asig.t_stop.rescale('s'),
                                1/eff_logMUA_rate) * asig.t_start.units

    non_nan_channels = [i for i in range(asig.shape[-1]) if np.isfinite(asig[:,i]).all()]
    asig_channels = asig[:, non_nan_channels]
    logMUA_signal = np.zeros((len(subsample_times), len(non_nan_channels)))

    for i, t in enumerate(subsample_times):
        if t < asig.t_start.rescale('s') + fft_slice/2:
            t_start = asig.t_start.rescale('s')
        elif t > asig.t_stop.rescale('s') - fft_slice/2:
            t_start = asig.t_stop.rescale('s') - fft_slice
        else:
            t_start = t - fft_slice/2

        t_stop = np.min([t_start + fft_slice,
                         asig.t_stop.rescale('s')]) *pq.s

        asig_slice = asig_channels.time_slice(t_start=t_start, t_stop=t_stop)

        freqs, psd = welch_psd(asig_slice,
                               frequency_resolution=highpass_frequency,
                               overlap=psd_overlap,
                               window='hann',
                               detrend='linear',
                               nfft=None)

        high_idx = (np.abs(freqs - lowpass_frequency)).argmin()
        if not i:
            print("logMUA signal estimated in frequency range "\
                + "{:.2f} - {:.2f}.".format(freqs[1], freqs[high_idx]))

        avg_power = np.mean(psd, axis=-1)
        # psd[:,1] corresponds to the highpass freq because that's the resolution
        avg_power_in_freq_band = np.mean(psd[:,1:high_idx], axis=-1)
        logMUA_signal[i] = np.squeeze(np.log(avg_power_in_freq_band/avg_power))

    new_signals = np.empty((len(subsample_times), channel_num))
    new_signals.fill(np.nan)
    new_signals[:, non_nan_channels] = logMUA_signal

    logMUA_asig = asig.duplicate_with_new_data(new_signals)
    logMUA_asig.array_annotations = asig.array_annotations
    logMUA_asig.sampling_rate = eff_logMUA_rate
    logMUA_asig.annotate(freq_band = [highpass_frequency, lowpass_frequency],
                         psd_frequency_resolution = highpass_frequency,
                         psd_overlap = psd_overlap,
                         psd_fs = fs)
    return logMUA_asig


def plot_logMUA_estimation(asig, logMUA_asig, highpass_frequency, lowpass_frequency,
                           t_start, t_stop, channel):
    asig = time_slice(asig, t_start, t_stop)
    logMUA_asig = time_slice(logMUA_asig, t_start, t_stop)
    filt_asig = butter(asig, highpass_frequency=highpass_frequency,
                             lowpass_frequency=lowpass_frequency)

    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    ax.plot(asig.times,
            zscore(asig.as_array()[:,channel]),
            label='original signal')

    ax.plot(filt_asig.times,
            zscore(filt_asig.as_array()[:,channel]) + 10,
            label=f'signal [{highpass_frequency}-{lowpass_frequency}]',
            alpha=0.5)

    ax.plot(logMUA_asig.times,
            zscore(logMUA_asig.as_array()[:,channel]) + 20,
            label='logMUA')

    ax.set_title('Channel {}'.format(channel))
    ax.set_xlabel('time [{}]'.format(asig.times.units.dimensionality.string))
    ax.set_yticklabels([])
    plt.legend()
    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    logMUA_rate = None if args.logMUA_rate is None \
                  else args.logMUA_rate*pq.Hz

    fft_slice = None if args.fft_slice is None \
                else args.fft_slice*pq.s

    asig = logMUA_estimation(block.segments[0].analogsignals[0],
                             highpass_frequency=args.highpass_frequency*pq.Hz,
                             lowpass_frequency=args.lowpass_frequency*pq.Hz,
                             logMUA_rate=logMUA_rate,
                             psd_overlap=args.psd_overlap,
                             fft_slice=fft_slice)

    if args.plot_channels is not None:
        for channel in args.plot_channels:
            plot_logMUA_estimation(asig=block.segments[0].analogsignals[0],
                                   logMUA_asig=asig,
                                   highpass_frequency=args.highpass_frequency*pq.Hz,
                                   lowpass_frequency=args.lowpass_frequency*pq.Hz,
                                   t_start=args.plot_tstart,
                                   t_stop=args.plot_tstop,
                                   channel=channel)
            output_path = args.img_dir \
                        / args.img_name.replace('_channel0', f'_channel{channel}')
            save_plot(output_path)

    asig.description += "Estimated logMUA signal [{}, {}] Hz ({}). "\
                        .format(args.highpass_frequency, args.lowpass_frequency,
                                os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
