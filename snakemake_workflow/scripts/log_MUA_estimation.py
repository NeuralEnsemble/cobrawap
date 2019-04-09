import numpy as np
import elephant as el
import neo
import argparse
from load_and_transform_to_neo import load_segment, save_segment


def detrending(asig, order):
    # ToDo: Improve algorithm and include into elephant
    X = asig.as_array()
    window_size = len(asig)
    if order > 0:
        X = X - np.mean(X, axis=0)
    if order > 1:
        factor = [1, 1/2., 1/6.]
        for i in np.arange(order-1)+1:
            detrend = np.linspace(-window_size/2., window_size/2., window_size)**i \
                      * np.mean(np.diff(X, n=i, axis=0)) * factor[i-1]
            X = X - detrend
    return X


def build_logMUA_segment(segment, freq_band, detrending_order, psd_freq_res, psd_overlap):
    asig = segment.analogsignals[0]

    fs = asig.sampling_rate.rescale('1/s').magnitude
    FFTWindowSize = int(round(fs / freq_band[0]))
    sample_num = int(np.floor(len(asig)/FFTWindowSize))
    MUA_sampling_rate = sample_num / (asig.t_stop - asig.t_start)

    logMUA_segment = neo.core.Segment()

    for asig in segment.analogsignals:
        logMUA_asig = logMUA_estimation(asig, fs, sample_num, FFTWindowSize, freq_band,
                                        MUA_sampling_rate, detrending_order,
                                        psd_freq_res, psd_overlap)
        logMUA_segment.analogsignals.append(logMUA_asig)

    return logMUA_segment


def logMUA_estimation(analogsignal, fs, sample_num, FFTWindowSize, freq_band,
                      MUA_sampling_rate, detrending_order, psd_freq_res, psd_overlap):
    MUA = np.zeros(sample_num)
    # calculating mean spectral power in each window
    for i in range(sample_num):
        local_asig = analogsignal[i*FFTWindowSize:(i+1)*FFTWindowSize]
        local_asig = detrending(local_asig, detrending_order)
        (f, p) = el.spectral.welch_psd(np.squeeze(local_asig),
                                       freq_res=psd_freq_res, overlap=psd_overlap,
                                       window='hanning', nfft=None, fs=fs,
                                       detrend=False, return_onesided=True,
                                       scaling='density', axis=-1)
        low_idx = np.where(freq_band[0] <= f)[0][0]
        high_idx = np.where(freq_band[1] <= f)[0][0]
        MUA[i] = np.mean(p[low_idx:high_idx])
    # ToDo: Fix bug that loading routine works with 'dimensionless' units
    logMUA_asig = neo.core.AnalogSignal(np.log(MUA), units='mV', t_start=analogsignal.t_start,
                                        t_stop=analogsignal.t_stop, sampling_rate=MUA_sampling_rate)
    logMUA_asig.annotations = analogsignal.annotations
    return logMUA_asig

    # ToDo: Normalization with basline power?


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs=1, type=str)
    CLI.add_argument("--data",      nargs=1, type=str)
    CLI.add_argument("--freq_band",  nargs=2, type=float)
    CLI.add_argument("--detrending_order", nargs=1, type=int, default=2)
    CLI.add_argument("--psd_freq_res",  nargs=1, type=int)
    CLI.add_argument("--psd_overlap",  nargs=1, type=float)

    args = CLI.parse_args()

    segment = load_segment(args.data[0])

    logMUA_segment = build_logMUA_segment(segment,
                                          freq_band=args.freq_band,
                                          detrending_order=args.detrending_order[0],
                                          psd_freq_res=args.psd_freq_res[0],
                                          psd_overlap=args.psd_overlap[0])
    save_segment(logMUA_segment, args.output[0])
