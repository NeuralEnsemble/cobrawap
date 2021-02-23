"""
Estimates the original activity by applying a digital filter with a given (protein specific) kernel function.
"""
import argparse
import quantities as pq
import os
import neo
import numpy as np
from scipy.signal import deconvolve
from utils import load_neo, write_neo, none_or_str, save_plot, none_or_float, \
                  AnalogSignal2ImageSequence, ImageSequence2AnalogSignal

def lognormal(t,mu,s):
    t_step = t[-1]/t.size
    log = 1/(t/t_step*np.sqrt(2*np.pi)*s)*np.exp(-(np.log(t/t_step)-mu)**2/(2*s**2))
    return log / (np.sum(log)*t_step) #normalized to the integral

def comb_exp(t,tr,td):
    t_step = t[-1]/t.size
    func = (1-np.exp(-t/tr))*(np.exp(-t/td))
    return func / (np.sum(func)*t_step) #normalized to the integral

def alpha(t,n,tau):
    t_step = t[-1]/t.size
    func = (t/tau)**n * np.exp(-t/tau)
    return func / (np.sum(func)*t_step) #normalized to the integral

def kernel_selection(kernel, params, time):
    if kernel == 'lognormal':
        ker = lognormal(time, params[0], params[1])

    elif kernel == 'combination_exp':
        ker = comb_exp(time, params[0], params[1])

    elif kernel == 'alpha':
        ker = alpha(time, params[0], params[1])

    else: #ToDo: a proper interruption?
        print("Kernel not valid!/n Default kernel: lognormal\n")
        ker = lognormal(time, 2.2, 0.91)

    return ker

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--kernel", nargs='?', type=str,
                     help="kernel response function", default="lognormal")
    CLI.add_argument("--parameters", nargs=2, type=none_or_float,
                     help="kernel parameters", default=[2.2,0.91])

    args = CLI.parse_args()

    # loads and converts the neo block
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)

    # converts the imagesequence class in a 3D Numpy array
    imgseq = block.segments[0].imagesequences[-1]
    imgseq_array = imgseq.as_array()
    dim_t, dim_x, dim_y = imgseq_array.shape

    #time axis
    t_step = block.segments[0].analogsignals[0].sampling_period.magnitude.tolist()
    t_end = block.segments[0].analogsignals[0].shape[0] * t_step
    time = np.arange(t_step,t_end+t_step,t_step)

    # get the chosen kernel
    params = args.parameters
    if params[0] and params[1]:
        ker = kernel_selection(args.kernel, params, time)
    else: #ToDo: automatic parameters choice when None is recognized
        print("Functionality in development, please enter parameters manually.\n")
        params = [2.2, 0.91] #default parameters
        ker = kernel_selection('lognormal', params, time) #default kernel

    # ToDo: initial+final ramp, which will change dim_t

    # decovolution script
    twoseconds = int(round(2/t_step)) # index corresponding to 2s
    newdim_t = dim_t-twoseconds+1 # deconvolution samples : n-m+1
    activities = np.zeros((newdim_t,dim_x,dim_y))
    for i in range(dim_x):
        for j in range(dim_y):
            activities[:,i,j] = deconvolve(imgseq_array[:,i,j], ker[:twoseconds])[0]

    #ToDo: add kernel plot and original/processed normalized plot

    """
    Old method, it doesn't work properly (the ouput block doesn't work as input for the other pipeline modules)
    
    # re-converting into analogsignal
    signal = activity.reshape((dim_t, dim_x * dim_y))
    asig = block.segments[0].analogsignals[0].duplicate_with_new_data(signal)
    asig.array_annotate(**block.segments[0].analogsignals[0].array_annotations)
    
    asig.name += ""
    asig.description += "Deconvoluted activity using the given {} kernel"\
                        .format(args.kernel)
    block.segments[0].analogsignals[0] = asig
    """
    # New method, creating a new block
    # create an ImageSequence with the deconvoluted matrix
    imgseq_deconv = neo.ImageSequence(activities, units = block.segments[0].analogsignals[0].units, 
                      sampling_rate = block.segments[0].analogsignals[0].sampling_rate, 
                      spatial_scale = block.segments[0].imagesequences[0].spatial_scale, 
                      name = block.segments[0].analogsignals[0].name, 
                      description = block.segments[0].analogsignals[0].description)

    # create a new Block & Segment and append the ImageSequence
    segm_deconv = neo.Segment()
    segm_deconv.annotations = block.segments[0].annotations
    segm_deconv.annotate(kernel = args.kernel) #ToDo: parameters annotations
    segm_deconv.imagesequences.append(imgseq_deconv)
    block_deconv = neo.Block()
    block_deconv.segments.append(segm_deconv)
    block_deconv.name = block.name
    block_deconv.description = block.description
    block_deconv.annotations = block.annotations

    # converting into analogsignal
    block_deconv = ImageSequence2AnalogSignal(block_deconv)

    write_neo(args.output, block_deconv)