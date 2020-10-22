import os
import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
import math
from utils import AnalogSignal2ImageSequence, load_neo, save_plot
from utils import load_neo, write_neo, remove_annotations
from utils import none_or_str, none_or_float
import neo

def ReadPixelData(events, DIM_X, DIM_Y):

    
    PixelLabel = events.array_annotations['y_coords'] * DIM_Y + events.array_annotations['x_coords']
    UpTrans = events.times
    
    Sorted_Idx = np.argsort(UpTrans)
    UpTrans = UpTrans[Sorted_Idx]
    PixelLabel = PixelLabel[Sorted_Idx]
    
    pixelID, counts = np.unique(PixelLabel, return_counts = True)
    nPixel=len(pixelID);

    nTrans = np.zeros(DIM_X*DIM_Y)
    nTrans[pixelID] = counts
    
    UpTrans_Evt = neo.Event(times=UpTrans,
                #labels = Label,
                name='UpTrans',
                array_annotations={'channels':PixelLabel},
                description='Transitions from down to up states. '\
                           +'Annotated with the channel id ("channels")',
                Dim_x = DIM_X,
                Dim_y = DIM_Y,
                spatial_scale = spatial_scale)

    remove_annotations(UpTrans_Evt, del_keys=['nix_name', 'neo_name'])
    UpTrans_Evt.annotations.update(UpTrans_Evt.annotations)
    
    return(UpTrans_Evt)

# ======================================================================================#

# LOAD input data
if __name__ == '__main__':
    
    CLI = argparse.ArgumentParser(description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                        help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                        help="path of output file")

    args = CLI.parse_args()
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[-1]
    DIM_X, DIM_Y = np.shape(imgseq[0])
    spatial_scale = imgseq.spatial_scale
    
    asig = block.segments[0].analogsignals[-1]
    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

    print('block 1 start', evts.annotations)
    
    UpTrans_Evt = ReadPixelData(evts, DIM_X, DIM_Y)
    
    block.segments[0].events.append(UpTrans_Evt)
    
    print('block 1 end', block.segments[0])

    write_neo(args.output, block)

