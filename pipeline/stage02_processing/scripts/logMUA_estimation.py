import numpy as np
from elephant.spectral import welch_psd
import quantities as pq
import argparse
import os
from utils import load_neo, write_neo, none_or_float


def logMUA_estimation(asig, highpass_freq, lowpass_freq, logMUA_rate,
                      psd_overlap, fft_slice):
    time_steps, channel_num = asig.shape
    fs = asig.sampling_rate.rescale('Hz')
    if fft_slice is None:
        fft_slice = (1/highpass_freq).rescale('s')
    elif fft_slice < 1/highpass_freq:
        raise InputError("Too few fft samples to estimate the frequency "\
                       + "content in the range [{} {}]Hz."\
                         . format(highpass_freq, lowpass_freq))
    # logMUA_rate can only be an int fraction of the orginal sampling_rate
    if logMUA_rate is None:
        logMUA_rate = highpass_freq
    if logMUA_rate > fs:
        raise InputError("The requested logMUA rate can not be larger than "\
                       + "the inital sampling rate!")
    subsample_order = int(fs/logMUA_rate)
    eff_logMUA_rate = fs/subsample_order
    print("effective logMUA rate = {} Hz".format(eff_logMUA_rate))

    if 1/eff_logMUA_rate > fft_slice:
        raise ValueError("The given logMUA_rate is too low to capture "\
                       + "the logMUA estimate of all the signal with "\
                       + "sample size {}. ".format(fft_slice)\
                       + "Either increase the logMUA_rate "\
                       + "or increase fft_slice.")

    subsample_times = np.arange(asig.t_start.rescale('s'),
                                asig.t_stop.rescale('s'),
                                1/eff_logMUA_rate) * asig.t_start.units

    logMUA_signal = np.zeros((len(subsample_times), channel_num))

    for i, t in enumerate(subsample_times):
        if t < asig.t_start.rescale('s') + fft_slice/2:
            t_start = asig.t_start.rescale('s')
        elif t > asig.t_stop.rescale('s') - fft_slice/2:
            t_start = asig.t_stop.rescale('s') - fft_slice
        else:
            t_start = t - fft_slice/2

        t_stop = np.min([t_start + fft_slice,
                         asig.t_stop.rescale('s')]) *pq.s

        freqs, psd = welch_psd(asig.time_slice(t_start=t_start,
                                               t_stop=t_stop),
                               freq_res=highpass_freq,
                               overlap=psd_overlap,
                               window='hanning',
                               detrend='linear',
                               nfft=None)

        high_idx = (np.abs(freqs - lowpass_freq)).argmin()
        if not i:
            print("logMUA signal estimated in frequency range "\
                + "{:.2f} - {:.2f} Hz.".format(freqs[1], freqs[high_idx]))

        avg_power = np.mean(psd, axis=-1)
        avg_power_in_freq_band = np.mean(psd[:,1:high_idx], axis=-1)
        logMUA_signal[i] = np.squeeze(np.log(avg_power_in_freq_band/
                                          avg_power))

    logMUA_asig = asig.duplicate_with_new_data(logMUA_signal)
    logMUA_asig.array_annotations = asig.array_annotations
    logMUA_asig.sampling_rate = eff_logMUA_rate
    logMUA_asig.annotate(freq_band = [highpass_freq, lowpass_freq],
                      psd_freq_res = highpass_freq,
                      psd_overlap = psd_overlap,
                      psd_fs = fs)
    return logMUA_asig


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--highpass_freq", nargs='?', type=float, required=True,
                     help="lower bound of frequency band in Hz")
    CLI.add_argument("--lowpass_freq", nargs='?', type=float, required=True,
                     help="upper bound of frequency band in Hz")
    CLI.add_argument("--logMUA_rate", nargs='?', type=none_or_float, default=None,
                     help="rate of the signal after transformation")
    CLI.add_argument("--psd_overlap", nargs='?', type=float, default=0.5,
                     help="overlap parameter for Welch's algorithm [0-1]")
    CLI.add_argument("--fft_slice",   nargs='?', type=none_or_float, default=None,
                     help="time window length used for power spectrum estimate, in s")
    args = CLI.parse_args()

    block = load_neo(args.data)

    logMUA_rate = None if args.logMUA_rate is None \
                  else args.logMUA_rate*pq.Hz

    fft_slice = None if args.fft_slice is None \
                else args.fft_slice*pq.s

    asig = logMUA_estimation(block.segments[0].analogsignals[0],
                          highpass_freq=args.highpass_freq*pq.Hz,
                          lowpass_freq=args.lowpass_freq*pq.Hz,
                          logMUA_rate=logMUA_rate,
                          psd_overlap=args.psd_overlap,
                          fft_slice=fft_slice)

    asig.name += ""
    asig.description += "Estimated logMUA signal [{}, {}] Hz ({}). "\
                        .format(args.highpass_freq, args.lowpass_freq,
                                os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    write_neo(args.output, block)
