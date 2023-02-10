import argparse
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
import neo
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from distutils.util import strtobool
from utils.io import load_neo, write_neo, save_plot
from utils.parse import none_or_str
from utils.neo_utils import imagesequence_to_analogsignal, analogsignal_to_imagesequence
from utils.convolve import phase_conv2d, get_kernel, nan_conv2d


def gradient_via_convolution(frames, kernelX, kernelY, are_phases=False):

    nan_channels = np.where(np.bitwise_not(np.isfinite(frames[0])))

    vector_frames = np.zeros(frames.shape, dtype=complex)

    for i, frame in enumerate(frames):
        if are_phases:
            dframe_x = phase_conv2d(frame, kernelX)
            dframe_y = phase_conv2d(frame, kernelY)
        else:
            dframe_x = nan_conv2d(frame, kernelX)
            dframe_y = nan_conv2d(frame, kernelY)
        dframe = (-1)*dframe_x - 1j*dframe_y

        vector_frames[i] = dframe
        vector_frames[i][nan_channels] = np.nan + np.nan*1j

    frames[:,nan_channels[0],nan_channels[1]] = np.nan
    return vector_frames


def smooth_frames(frames, sigma):
    # replace nan sites by median
    if np.isfinite(frames).any():
        # assume constant nan sites over time
        nan_sites = np.where(np.bitwise_not(np.isfinite(frames[0])))
        if np.iscomplexobj(frames):
            frames[:,nan_sites[0],nan_sites[1]] = np.nanmedian(np.real(frames)) \
                                                + np.nanmedian(np.imag(frames))*1j
        else:
            frames[:,nan_sites[0],nan_sites[1]] = np.nanmedian(frames)
    else:
        nan_sites = None

    # apply gaussian filter
    if np.iscomplexobj(frames):
        frames = gaussian_filter(np.real(frames), sigma=sigma, mode='nearest') \
               + gaussian_filter(np.imag(frames), sigma=sigma, mode='nearest')*1j
    else:
        frames = gaussian_filter(frames, sigma=sigma, mode='nearest')

    # set nan sites back to nan
    if nan_sites is not None:
        if np.iscomplexobj(frames):
            frames[:,nan_sites[0],nan_sites[1]] = np.nan + np.nan*1j
        else:
            frames[:,nan_sites[0],nan_sites[1]] = np.nan

    return frames


def plot_gradient_via_convolution(frame, vec_frame, skip_step=None, are_phases=False):
    # Every <skip_step> point in each direction.
    fig, ax = plt.subplots()
    dim_y, dim_x = vec_frame.shape
    if are_phases:
        cmap = 'twilight'
        vmin, vmax = -np.pi, np.pi
    else:
        cmap = 'viridis'
        vmin, vmax = None, None
    img = ax.imshow(frame, interpolation='nearest', vmin=vmin, vmax=vmax,
                    cmap=plt.get_cmap(cmap), origin='lower')
    plt.colorbar(img, ax=ax)
    if skip_step is None:
        skip_step = int(min([dim_x, dim_y]) / 50) + 1

    ax.quiver(np.arange(dim_x)[::skip_step],
              np.arange(dim_y)[::skip_step],
              np.real(vec_frame[::skip_step,::skip_step]),
              np.imag(vec_frame[::skip_step,::skip_step]))

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def is_phase_signal(signal, use_phases):
    vmin = np.nanmin(signal)
    vmax = np.nanmax(signal)
    in_range = np.isclose(np.array([vmin,   vmax]),
                          np.array([-np.pi, np.pi]), atol=0.05)
    if in_range.all():
        print('The signal values seem to be phase values [-pi, pi]!')
        print(f'The setting "use_phases" is {bool(use_phases)}.')
        if use_phases:
            print('Thus, the phase transformation is skipped.')
        else:
            print('Anyhow, the signal is treated as phase signal in the following. If this is not desired please review the preprocessing.')
        return True
    else:
        return False


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--gaussian_sigma", nargs='+', type=float, default=[0,3,3],
                     help='sigma of gaussian filter in each dimension')
    CLI.add_argument("--derivative_filter", nargs='?', type=none_or_str, default=None,
                     help='Filter kernel to use for calculating spatial derivatives')
    CLI.add_argument("--use_phases", nargs='?', type=strtobool, default=False,
                     help='whether to use signal phase instead of amplitude')
    args, unknown = CLI.parse_known_args()
    block = load_neo(args.data)

    asig = block.segments[0].analogsignals[0]
    imgseq = analogsignal_to_imagesequence(asig)

    frames = imgseq.as_array()
    # frames /= np.nanmax(np.abs(frames))

    if is_phase_signal(frames, args.use_phases):
        args.use_phases = True
    elif args.use_phases:
        analytic_frames = hilbert(frames, axis=0)
        frames = np.angle(analytic_frames)

    kernel = get_kernel(args.derivative_filter)

    vector_frames = gradient_via_convolution(frames=frames,
                                             kernelX=kernel.x,
                                             kernelY=kernel.y,
                                             are_phases=args.use_phases)

    if np.sum(args.gaussian_sigma):
        vector_frames = smooth_frames(vector_frames, sigma=args.gaussian_sigma)

    vec_imgseq = neo.ImageSequence(vector_frames,
                                   units='dimensionless',
                                   dtype=complex,
                                   t_start=imgseq.t_start,
                                   spatial_scale=imgseq.spatial_scale,
                                   sampling_rate=imgseq.sampling_rate,
                                   name='gradient',
                                   description='Gradient estimation by kernel convolution.',
                                   kernel_type=args.derivative_filter,
                                   file_origin=imgseq.file_origin)
   
    vec_imgseq.annotations = copy(imgseq.annotations)

    if args.output_img is not None:
        ax = plot_gradient_via_convolution(frames[10], vector_frames[10],
                                           skip_step=None, are_phases=args.use_phases)
        ax.set_ylabel(f'pixel size: {imgseq.spatial_scale} ')
        ax.set_xlabel('{:.3f} s'.format(asig.times[0].rescale('s').magnitude))
        save_plot(args.output_img)

    # block.segments[0].imagesequences = [vec_imgseq]
    vec_asig = imagesequence_to_analogsignal(vec_imgseq)

    block.segments[0].analogsignals.append(vec_asig)

    write_neo(args.output, block)
