import numpy as np
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
import neo
import quantities as pq
import argparse
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape


def MUA_estimation(asig, highpass_freq, lowpass_freq, MUA_rate, psd_overlap,
                   fft_samples):
    time_steps, channel_num = asig.shape
    fs = asig.sampling_rate.rescale('Hz')
    # fft_samples = int(np.round((fs/highpass_freq).magnitude))
    if fft_samples is None:
        fft_samples = int(np.round((fs/highpass_freq).magnitude))
    elif fft_samples < int(np.round((fs/highpass_freq).magnitude)):
        raise InputError("Too few fft samples to estimate the frequency "\
                       + "content in the range [{} {}]Hz."\
                         . format(highpass_freq, lowpass_freq))
    # MUA_rate can only be an int fraction of the orginal sampling_rate
    if MUA_rate is None:
        MUA_rate = highpass_freq
    if MUA_rate > fs:
        raise InputError("The requested MUA rate can not be larger than "\
                       + "the inital sampling rate!")
    subsample_order = int(np.round(fs/MUA_rate))
    eff_MUA_rate = fs/subsample_order
    print("effective MUA rate = {} Hz".format(eff_MUA_rate))

    if (fs/eff_MUA_rate).magnitude > fft_samples:
        raise ValueError("The given MUA_rate is too low to capture "\
                       + "the MUA estimate of all the signal with "\
                       + "sample size {}. ".format(fft_samples)\
                       + "Either increase the MUA_rate "\
                       + "or increase fft_samples.")

    subsample_idx = np.arange(0, len(asig.times), subsample_order)

    MUA_signal = np.zeros((len(subsample_idx), channel_num))

    for i, idx in enumerate(subsample_idx):
        start_idx = max([0, idx-int(fft_samples/2)])
        stop_idx = min([start_idx+fft_samples, time_steps-1])
        freqs, psd = welch_psd(asig.time_slice(t_start=start_idx/fs,
                                               t_stop=stop_idx/fs),
                               freq_res=highpass_freq,
                               overlap=psd_overlap,
                               window='hanning',
                               detrend='constant',
                               nfft=None)
        norm = np.median(psd, axis=-1)
        high_idx = (np.abs(freqs - lowpass_freq)).argmin()
        if high_idx < len(freqs)-1:
            high_idx += 1
        # ToDo: log !?
        MUA_signal[i] = np.squeeze(np.log(np.mean(psd[:,1:high_idx], axis=-1)/norm))

    MUA_asig = asig.duplicate_with_new_data(MUA_signal)
    MUA_asig.array_annotations = asig.array_annotations
    MUA_asig.sampling_rate = eff_MUA_rate
    MUA_asig.annotate(freq_band = [highpass_freq, lowpass_freq],
                      psd_freq_res = highpass_freq,
                      psd_overlap = psd_overlap,
                      psd_fs = fs)
    return MUA_asig


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",        nargs='?', type=str)
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--highpass_freq", nargs='?', type=float)
    CLI.add_argument("--lowpass_freq",  nargs='?', type=float)
    CLI.add_argument("--MUA_rate",      nargs='?', type=none_or_float)
    CLI.add_argument("--psd_overlap",   nargs='?', type=float)
    CLI.add_argument("--fft_samples",   nargs='?', type=none_or_int)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)

    if args.MUA_rate is None:
        MUA_rate = args.MUA_rate
    else:
        MUA_rate = args.MUA_rate*pq.Hz

    asig = block.segments[0].analogsignals[0]
    asig = MUA_estimation(asig,
                          highpass_freq=args.highpass_freq*pq.Hz,
                          lowpass_freq=args.lowpass_freq*pq.Hz,
                          MUA_rate=MUA_rate,
                          psd_overlap=args.psd_overlap,
                          fft_samples=args.fft_samples)

    # save processed data
    asig.name += ""
    asig.description += "Estimated MUA signal [{}, {}] Hz ({}). "\
                        .format(args.highpass_freq, args.lowpass_freq,
                                os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
