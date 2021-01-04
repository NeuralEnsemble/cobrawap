"""
Deconvolutes the given signal by applying a digital filter with a given kernel function.
"""
import argparse
import quantities as pq
import os
import matplotlib.pyplot as plt
import neo
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import *
from scipy.optimize import curve_fit
from utils import load_neo, write_neo, none_or_str, save_plot, none_or_float, \
                  AnalogSignal2ImageSequence, ImageSequence2AnalogSignal

def gauss(x, mu, sigma, a):
    return a/(2 * np.pi * sigma**2)**(1/2) * np.exp(- (x - mu)**2 / (2 * sigma**2))

def start_point(quantile,y,res):
    for x in range(y.size):
        if np.sum(y[:x])*res < np.sum(y)*res*quantile:
            start = x
    return start

def plot_KDE(kde_act, x_KDE, ker_pdf, kde_mod, popt, sp_res, start):
    fig, ax = plt.subplots()

    ax.hist(kde_act, density=True, bins='fd', alpha=0.5, label='Histogram')
    ax.set_title("KDE of the activity", fontsize=18)
    ax.plot(x_KDE, ker_pdf, color='red', label='KDE')
    ax.axvline(x=kde_mod, linestyle='--', color='black', label=f'Mode={kde_mod:.4E}')
    ax.plot(x_KDE, gauss(x_KDE, popt[0], popt[1], popt[2]),
            label="Gaussian fit", color='green')
    ax.axvline(x= kde_act.min() + sp_res*start, linestyle='--', color='purple', label="fit start value")
    ax.plot(np.arange(popt[0]-popt[1],kde_mod,1),[ker_pdf.max()/2]*np.arange(popt[0]-popt[1],kde_mod,1).size, color='blue',
            label=f"sigma={popt[1]:.3E}")
    ax.legend()
    ax.set_xlabel('Activity', fontsize=12)
    ax.set_ylabel('Probability density function', fontsize=12)

    return ax

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                         help="path of output image", default=None)
    CLI.add_argument("--bandwidth", nargs='?', type=str,
                     help="KDE bandwidth", default="silverman")
    CLI.add_argument("--quantile", nargs='?', type=float,
                     help="Quantile of the left outliers", default=0.05)
    # think about using alternatives to the gaussian kernel

    args = CLI.parse_args()

    # loads and converts the neo block
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    
    # converts the imagesequence class in a 3D Numpy array
    imgseq = block.segments[0].imagesequences[-1]
    imgseq_array = imgseq.as_array()
    dim_t, dim_x, dim_y = imgseq_array.shape

    KDE_results = np.zeros((5,dim_x,dim_y)) # mode (no error), mean and sigma with errors for every pixel
    sp_res = 1 # fixing the resolution in the activities space
    n_points = 20 #KDE fit sampling points
    """
    get KDE results for every pixel
    over 20 minutes of compiling time...
    """
    # focusing on a single (central) pixel
    x, y = round(dim_x/2), round(dim_y/2)
    kde_act = imgseq_array[:,x,y]
    kde = gaussian_kde(kde_act, bw_method=args.bandwidth) #get the KDE
    x_KDE = np.arange(kde_act.min(), kde_act.max(), sp_res) # get the activities grid
    ker_pdf = kde(x_KDE) # get the probability density function applying the KDE on the grid 
    kde_mod = kde_act.min() + sp_res*ker_pdf.argmax() # mode as max(pdf)
    #KDE_results[0,x,y] = kde_mod
    # focus on the left side of the distribution
    start = start_point(args.quantile, ker_pdf, sp_res) #from the quantile
    end = ker_pdf.argmax() #to the peak
    xres = int(round((x_KDE[end]-x_KDE[start])/n_points)) #KDE sampling resolution
    xdata = x_KDE[start:end:xres]
    ydata = ker_pdf[start:end:xres]
    # fit, constraining the peak of the gaussian to be equal to the mode of the distribution
    p0 = [kde_mod, kde_act.std(), 1] # starting parameters
    popt, pcov = curve_fit(gauss, xdata, ydata, p0=p0, bounds=([kde_mod-0.1, 0, 0], [kde_mod, np.inf, np.inf]))
    perr = np.sqrt(np.diag(pcov))
    #KDE_results[1,x,y], KDE_results[2,x,y] = popt[0], popt[1]
    #KDE_results[3,x,y], KDE_results[4,x,y] = perr[0], perr[1]

    plot_KDE(kde_act, x_KDE, ker_pdf, kde_mod, popt, sp_res, start)
    save_plot(args.output_img)

    #results in annotations, writing the (new) block
    block.segments[0].annotate(mode=kde_mod, mean=popt[0], sigma=popt[1])
    write_neo(args.output, block)