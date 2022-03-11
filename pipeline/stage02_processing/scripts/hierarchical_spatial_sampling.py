import neo
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import argparse
from utils.io import load_neo, write_neo, save_plot
from utils.parse import none_or_float, none_or_int, none_or_str
from utils.neo import analogsignals_to_imagesequences, imagesequences_to_analogsignals


def next_power_of_2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1

def ComputeCenterOfMass(s, scale):
    # compute the center of mass of a macropixel con nan values
    mean = np.nanmean(s, axis = 2)
    idx = np.where(~np.isnan(mean))
    x_cm = (np.mean(idx[0])+0.5)*scale
    y_cm = (np.mean(idx[1])+0.5)*scale
    if np.isnan(x_cm): x_cm = np.shape(mean)[0]/2
    if np.isnan(y_cm): y_cm = np.shape(mean)[1]/2
    return(x_cm, y_cm)

# List con x,y,L,flag,x_parent, y_parent, L_parent

def CheckCondition(coords, Input_image, method = 'shapiro'):
    #function to check whether node is compliant with the condition
    value = np.nanmean(Input_image[coords[0]:coords[0]+coords[2], coords[1]:coords[1]+coords[2]], axis = (0,1))
    if np.isnan(np.nanmax(value)):
        return(1)
    else:
        if method == 'shapiro':
            stat, p = shapiro(value)
            if p <= 0.05:
                return(0)
            else:
                return(1)

def NewLayer(l, Input_image, evaluation_method):
    new_list = []
    # first leaf
    cond = CheckCondition([l[0], l[1], l[2]//2], Input_image, evaluation_method)
    new_list.append([l[0], l[1], l[2]//2, (l[3]+cond)*cond, l[0], l[1], l[2]])

    # second leaf
    cond = CheckCondition([l[0], l[1]+l[2]//2, l[2]//2], Input_image, evaluation_method)
    new_list.append([l[0], l[1]+l[2]//2, l[2]//2, (l[3]+cond)*cond, l[0], l[1], l[2]])

    # third leaf
    cond = CheckCondition([l[0]+l[2]//2, l[1], l[2]//2], Input_image, evaluation_method)
    new_list.append([l[0]+l[2]//2, l[1], l[2]//2, (l[3]+cond)*cond, l[0], l[1], l[2]])
    
    # fourth leaf
    cond = CheckCondition([l[0]+l[2]//2, l[1]+l[2]//2, l[2]//2], Input_image, evaluation_method)
    new_list.append([l[0]+l[2]//2, l[1]+l[2]//2, l[2]//2, (l[3]+cond)*cond, l[0], l[1], l[2]])
    
    return(new_list)

def CreateMacroPixel(Input_image, exit_method = 'consecutive', signal_eval_method = 'shapiro', threshold = 0.5, n_bad = 2):
    # initialized node list
    NodeList = []
    MacroPixelCoords = []

    # initialized root
    NodeList.append([0,0,np.shape(Input_image)[0], 0, 0,0,np.shape(Input_image)[0]])

    while len(NodeList):

        # create node's children
        Children = NewLayer(NodeList[0], Input_image, evaluation_method = signal_eval_method)
        NodeList.pop(0) # delete investigated node
       
        #check wether exit condition is met
        if exit_method == 'voting':
            # check how many of children are "bad"
            flag_list = [np.int32(f[3]>= n_bad) for f in Children]
            if np.sum(flag_list) > threshold*len(Children):
                MacroPixelCoords.append(Children[0][4:]) # store parent node
                Children = []
        else:
            # check if some or all children are "bad"
            flag_list = [f[3]==n_bad for f in Children]
            if all(flag_list): # if all children are "bad"
                MacroPixelCoords.append(Children[0][4:]) # store parent node
                Children = []
            else:
                Children = [ch for ch in Children if ch[3] != n_bad]

        # check whether minimum dimension has been reached
        l_list = [ch[2] == 1 for ch in Children]
        idx = np.where(l_list)[0]
        if len(idx):
            for i in range(0, len(l_list)):
                if l_list[i] == True:
                    MacroPixelCoords.append(Children[i][0:3])
            Children = [ch for ch in Children if ch[2] != 1]

        NodeList.extend(Children)

    return(MacroPixelCoords)


def plot_masked_image(original_img, MacroPixelCoords):

    NewImage = np.empty([np.shape(original_img)[0], np.shape(original_img)[0]])*np.nan
    for macro in MacroPixelCoords:
        # fill pixels belonging to the same macropixel with the same signal
        NewImage[macro[0]:macro[0] + macro[2], macro[1]:macro[1] + macro[2]] = np.mean(np.nanmean(original_img[macro[0]:macro[0] + macro[2], macro[1]:macro[1]+macro[2]], axis = (0,1)))

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(6,2, forward=True)
    im = axs[0].imshow(np.nanmean(original_img, axis = 2))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('Original image', fontsize = 7.)

    im = axs[1].imshow(NewImage)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('Post sampling', fontsize = 7.)

    ls = [macro[2] for macro in MacroPixelCoords]

    im = axs[2].hist(ls, bins = np.max(ls))
    axs[2].set_yscale('Log')
    axs[2].set_xlabel('macro-pixel size', fontsize = 7.)

    plt.tight_layout()
    return axs

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                     help="path of output image", default=None)
    CLI.add_argument("--n_bad_nodes",  nargs='?', type=none_or_int,
                     help="number of non informative macro-pixel to prune branch", default=2)
    CLI.add_argument("--exit_condition",  nargs='?', type=none_or_str,
                     help="exit condition in the optimal macro-pixel dimension tree search", default='consecutive')
    CLI.add_argument("--signal_eval_method",  nargs='?', type=none_or_str,
                     help="signal to noise ratio evalutaion method", default='shapiro')
    CLI.add_argument("--voting_threshold",  nargs='?', type=none_or_float,
                     help="threshold of non informative nodes percentage if voting method is selected", default=0.5)
    CLI.add_argument("--output_array",  nargs='?', type=none_or_str,
                      help="path of output numpy array", default=None)
    args = CLI.parse_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    block = analogsignals_to_imagesequences(block)
    
    # load image sequences at the original spatial resolution
    imgseq = block.segments[0].imagesequences[-1]
    imgseq_array = imgseq.as_array().T
    dim_x, dim_y, dim_t = imgseq_array.shape

    # pad image sequences with nans to make it divisible by 2
    N_pad = next_power_of_2(max(dim_x, dim_y)) 
    padded_image_seq = np.pad(imgseq_array, 
                         pad_width = [((N_pad-dim_x)//2,(N_pad-dim_x)//2 + (N_pad-dim_x)%2), 
                                      ((N_pad-dim_y)//2, (N_pad-dim_y)//2 + (N_pad-dim_y)%2), 
                                      (0,0)], mode = 'constant', constant_values = np.nan)
    # tree search for the best macro-pixel dimension
    # List con x,y,L,flag,x_parent, y_parent, L_parent
    MacroPixelCoords = CreateMacroPixel(Input_image = padded_image_seq,
                                        exit_method = args.exit_condition,
                                        signal_eval_method = args.signal_eval_method,
                                        threshold = args.voting_threshold,
                                        n_bad = args.n_bad_nodes)
    if args.output_img is not None:
        plot_masked_image(padded_image_seq, MacroPixelCoords)
        save_plot(args.output_img)

    signal = np.empty([len(MacroPixelCoords), dim_t]) #save data as analogsignal
    coordinates = np.empty([len(MacroPixelCoords), 3]) #pixel coordinates [x,y,L] to retrieve 
                                                       # original one
    ch_id = np.empty([len(MacroPixelCoords)]) # new channel id
    x_coord = np.empty([len(MacroPixelCoords)]) # new x coord
    y_coord = np.empty([len(MacroPixelCoords)]) # new y coord
    x_coord_cm = np.empty([len(MacroPixelCoords)]) # new x coord
    y_coord_cm = np.empty([len(MacroPixelCoords)]) # new y coord

    for px_idx, px in enumerate(MacroPixelCoords): # for each new pixel
        signal[px_idx, :] = np.nanmean(padded_image_seq[px[0]:px[0]+px[2], 
                                                     px[1]:px[1]+px[2]], 
                                    axis = (0,1))
        x_coord_cm[px_idx], y_coord_cm[px_idx] = ComputeCenterOfMass(padded_image_seq[px[0]:px[0]+px[2], px[1]:px[1]+px[2]],
                                 imgseq.spatial_scale)

        coordinates[px_idx] = px
        ch_id[px_idx] = px_idx
        x_coord[px_idx] = (px[0] + px[2]/2.)*imgseq.spatial_scale
        y_coord[px_idx] = (px[1] + px[2]/2.)*imgseq.spatial_scale

    new_evt_ann = {'x_coords': coordinates.T[0],
                   'y_coords': coordinates.T[1],
                   'x_coord_cm': x_coord_cm,
                   'y_coord_cm': y_coord_cm,
                   'pixel_coordinates_L': coordinates.T[2],
                   'channel_id': ch_id}


    new_asig = asig.duplicate_with_new_data(signal.T)
    #new_asig.array_annotations = asig.array_annotations
    new_asig.array_annotations.update(new_evt_ann)
    new_asig.name += ""
    new_asig.description += "Non homogeneous downsampling obtained by cheching the signal to noise ratio of macropixels ad different size."
    block.segments[0].analogsignals[0] = new_asig

    write_neo(args.output, block)



