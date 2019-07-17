import neo
import numpy as np
import skimage as sk
import quantities as pq
import argparse
import os
import matplotlib.pyplot as plt
import elephant as el


def flatten_image_focus(images):
    pixel = np.where(np.bitwise_not(np.isnan(images[0])))
    image_focus = np.zeros((images.shape[0], len(pixel[0])))
    for num, frame in enumerate(images):
        image_focus[num] = frame[pixel[0].tolist(),pixel[1].tolist()]
    return image_focus


if __name__ == '__main__':

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--image_file", nargs='?', type=str)
    CLI.add_argument("--out_spectrum", nargs='?', type=str)
    CLI.add_argument("--out_plot", nargs='?', type=str)
    CLI.add_argument("--sampling_rate", nargs='?', type=float)
    CLI.add_argument("--psd_freq_res", nargs='?', type=float)
    CLI.add_argument("--psd_overlap", nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.image_file) as io:
        images = io.read_block().segments[0].analogsignals[0]

    image_focus = flatten_image_focus(images)

    freqs, psd = el.spectral.welch_psd(image_focus.T,
                                       fs=args.sampling_rate,
                                       freq_res=args.psd_freq_res*pq.Hz,
                                       overlap=args.psd_overlap)

    spectrum = np.zeros((2,len(freqs)))
    spectrum[0] = freqs
    spectrum[1] = np.mean(psd, axis=0)

    np.save(args.out_spectrum, spectrum)

    fig, ax = plt.subplots()
    ax.plot(spectrum[0], spectrum[1])
    ax.set_title('Average power spectrum')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('Fourier signal')
    plt.savefig(args.out_plot)
