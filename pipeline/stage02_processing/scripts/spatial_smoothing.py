"""
Substract the background of a given dataset by substracting the mean of each channel.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import quantities as pq
from skimage import data, io, filters, measure
from utils import determine_spatial_scale, load_neo, write_neo, save_plot, \
                  none_or_str


def spatial_smoothing(asig, MACRO_PIXEL_DIM, DIM_X, DIM_Y):

    print("         Spacial smoothing...")
    # Now we need to reduce the noise from the images by performing a spatial smoothing
    img_collection_reduced = []
    for elem in range(0, len(asig)): #for each image
        img = np.reshape(asig[elem], (int(DIM_X),int(DIM_Y)))
        img_reduced = measure.block_reduce(img, (MACRO_PIXEL_DIM, MACRO_PIXEL_DIM), np.mean)
        img_reduced = np.reshape(img_reduced,(1, len(img_reduced)*len(img_reduced)))
        img_collection_reduced.append(img_reduced[0,:])
    print("         Images reduced")

    #----------------------
    return img_collection_reduced



if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                     help="path of output image", default=None)
    CLI.add_argument("--output_array",  nargs='?', type=none_or_str,
                      help="path of output numpy array", default=None)

    CLI.add_argument("--macro_pixel_dim",  nargs='?', type=int,
                      help="smoothing factor", default=2)
    CLI.add_argument("--dim_X",  nargs='?', type=int,
                      help="original x dimension", default=100)
    CLI.add_argument("--dim_Y",  nargs='?', type=int,
                      help="original y dimension", default=100)
    
    
    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    
    MACRO_PIXEL_DIM = args.macro_pixel_dim
    DIM_X = args.dim_X
    DIM_Y = args.dim_Y

    spatial_scale_pre = asig.annotations['spatial_scale']
    
    img_reduced = spatial_smoothing(asig, MACRO_PIXEL_DIM, DIM_X, DIM_Y)

    if args.output_img or args.output_array is not None:
        if args.output_array is not None:
            np.save(args.output_array, img_reduced[0])
        if args.output_img is not None:
            plt.figure()
            plt.imshow(np.reshape(img_reduced[0], (int(DIM_X/MACRO_PIXEL_DIM),int(DIM_Y/MACRO_PIXEL_DIM))))
            
            save_plot(args.output_img)

    asig = block.segments[0].analogsignals[0].duplicate_with_new_data(img_reduced)
    x_c = np.zeros(len(img_reduced[0]))
    y_c = np.zeros(len(img_reduced[0]))
    
    for x in range(0, int(DIM_Y/MACRO_PIXEL_DIM)):
        y_c[int(DIM_Y/MACRO_PIXEL_DIM)*x:int(DIM_Y/MACRO_PIXEL_DIM)*(x+1)] = x
        x_c[int(DIM_X/MACRO_PIXEL_DIM)*x:int(DIM_X/MACRO_PIXEL_DIM)*(x+1)] = list(range(0,int(DIM_X/MACRO_PIXEL_DIM)))
        
    asig.array_annotations = {'x_coords': x_c, 'y_coords': y_c}
    asig.annotations = {'spatial_scale': spatial_scale_pre.magnitude * MACRO_PIXEL_DIM * pq.mm}

    asig.name += ""
    asig.description += "The spatial dim reduced ({})."\
                        .format(os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig
    
    write_neo(args.output, block)
