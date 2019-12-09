import numpy as np
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
import neo
import quantities as pq
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'../'))
from utils import check_analogsignal_shape


def MUA_estimation(asig, highpass_freq, lowpass_freq, MUA_rate, psd_overlap,
                   fft_slice):
    time_steps, channel_num = asig.shape
    fs = asig.sampling_rate.rescale('Hz')
    if fft_slice is None:
        fft_slice = (1/highpass_freq).rescale('s')
    elif fft_slice < 1/highpass_freq:
        raise InputError("Too few fft samples to estimate the frequency "\
                       + "content in the range [{} {}]Hz."\
                         . format(highpass_freq, lowpass_freq))
    # MUA_rate can only be an int fraction of the orginal sampling_rate
    if MUA_rate is None:
        MUA_rate = highpass_freq
    if MUA_rate > fs:
        raise InputError("The requested MUA rate can not be larger than "\
                       + "the inital sampling rate!")
    subsample_order = int(fs/MUA_rate)
    eff_MUA_rate = fs/subsample_order
    print("effective MUA rate = {} Hz".format(eff_MUA_rate))

    if 1/eff_MUA_rate > fft_slice:
        raise ValueError("The given MUA_rate is too low to capture "\
                       + "the MUA estimate of all the signal with "\
                       + "sample size {}. ".format(fft_slice)\
                       + "Either increase the MUA_rate "\
                       + "or increase fft_slice.")

    subsample_times = np.arange(asig.t_start.rescale('s'),
                                asig.t_stop.rescale('s'),
                                1/eff_MUA_rate) * asig.t_start.units

    MUA_signal = np.zeros((len(subsample_times), channel_num))

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
        # ToDo: log !?
        if not i:
            print("MUA signal estimated in frequency range "\
                + "{:.2f} - {:.2f} Hz.".format(freqs[1], freqs[high_idx]))

        if t_stop < 6*pq.s:
            fig, ax = plt.subplots(ncols=2)
            current_signal = asig.time_slice(t_start=asig.t_start, t_stop=6*pq.s)
            palette = sns.color_palette("Set2")
            for j, channel_psd in enumerate(psd[:8]):
                ax[0].plot(freqs, channel_psd, c=palette[j])
                ax[1].plot(current_signal.times,
                           current_signal.as_array()[:,j],
                           color=palette[j])
                ax[1].axvspan(t_start, t_stop, alpha=0.2, color='k')
            ax[1].set_title('{:.3f} - {:.3f}s'.format(t_start, t_stop))
            ax[1].set_xlabel('time [s]')
            ax[0].set_xlabel('frequency [Hz]')
            ax[0].set_ylabel('spectral density')
            ax[0].set_ylim((0,3*10**-7))
            frame_folder = '/home/rgutzen/ProjectsData/wave_analysis_pipeline/stage03_trigger_detection/transformation/MUA_estimation/MUA_estimation_frames'
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)
            plt.savefig(os.path.join(frame_folder, 'frame_{}.png'.format(str(i).zfill(5))))
            plt.close(fig)

        avg_power = np.mean(psd, axis=-1)
        avg_power_in_freq_band = np.mean(psd[:,1:high_idx], axis=-1)
        MUA_signal[i] = np.squeeze(np.log(avg_power_in_freq_band/
                                          avg_power))

    MUA_asig = asig.duplicate_with_new_data(MUA_signal)
    MUA_asig.array_annotations = asig.array_annotations
    MUA_asig.sampling_rate = eff_MUA_rate
    MUA_asig.annotate(freq_band = [highpass_freq, lowpass_freq],
                      psd_freq_res = highpass_freq,
                      psd_overlap = psd_overlap,
                      psd_fs = fs)
    return MUA_asig


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
    CLI.add_argument("--fft_slice",   nargs='?', type=none_or_float)
    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        block = io.read_block()

    check_analogsignal_shape(block.segments[0].analogsignals)

    MUA_rate = None if args.MUA_rate is None \
               else args.MUA_rate*pq.Hz

    fft_slice = None if args.fft_slice is None \
                else args.fft_slice*pq.s


    asig = block.segments[0].analogsignals[0]
    asig = MUA_estimation(asig,
                          highpass_freq=args.highpass_freq*pq.Hz,
                          lowpass_freq=args.lowpass_freq*pq.Hz,
                          MUA_rate=MUA_rate,
                          psd_overlap=args.psd_overlap,
                          fft_slice=fft_slice)

    # save processed data
    asig.name += ""
    asig.description += "Estimated MUA signal [{}, {}] Hz ({}). "\
                        .format(args.highpass_freq, args.lowpass_freq,
                                os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig

    with neo.NixIO(args.output) as io:
        io.write(block)
