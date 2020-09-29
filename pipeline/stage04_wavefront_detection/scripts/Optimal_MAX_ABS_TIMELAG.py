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


def Optimal_MAX_ABS_TIMELAG(nTrans, UpTrans, ChLabel, MAX_ABS_TIMELAG):

    ####################################################
    # ExpectedTrans i.e. estimate of the Number of Waves
    # ExpectedTrans used to estimate/optimize IWI
    eExpectedTrans = np.zeros(3)
    TransPerCh = nTrans;
    eExpectedTrans[0] = np.median(TransPerCh[np.where(TransPerCh != 0)]);
    Active = np.sort(TransPerCh[np.where(TransPerCh != 0)]);
    eExpectedTrans[1] = np.median(Active[len(Active)-8:len(Active)]); #consider THE 8 MOST ACTIVE CHANNELS only
    eExpectedTrans[2] = np.mean(TransPerCh[np.where(TransPerCh != 0)])+np.std(TransPerCh[np.where(TransPerCh != 0)]);
    sel1=0;
    ExpectedTrans=eExpectedTrans[sel1];

    print('ExpectedTrans:', ExpectedTrans)
    
    ####################################################
    # Compute the time lags matrix looking for optimal MAX_ABS_TIMELAG...
    # (depends on the distance between electrodes in the array)
    
    DeltaTL = np.diff(UpTrans);

    OneMoreLoop = 1;
    iter_macro = 0
    while OneMoreLoop==1:
        WnW = DeltaTL<=MAX_ABS_TIMELAG;
        ndxBegin = np.where(WnW==1)[0][0];
        nw = -1;
        
        Wave = []
        WaveUnique = []
        WaveSize = []
        WaveTime = []
        while ndxBegin < len(DeltaTL):
            try:
                ndxEnd = np.where(WnW[ndxBegin:len(WnW)]==0)[0][0]+ndxBegin; # isolated transitions are EXCLUDED
            except IndexError:
                ndxEnd = len(UpTrans)-1;
            
            nw = nw + 1;
            
            ndx = list(range(ndxBegin,ndxEnd+1))
            
            Wave.append({'ndx': list(range(ndxBegin,ndxEnd+1))});
            WaveUnique.append(len(ndx) == len(np.unique(ChLabel[ndx])));
            WaveSize.append(len(ndx));
            WaveTime.append(np.mean(UpTrans[ndx]));

            if ndxEnd == len(UpTrans)-1:
                ndxBegin = len(DeltaTL);
            else:
                try:
                    ndxBegin = np.where(WnW[ndxEnd:len(WnW)]==1)[0][0]+ndxEnd;
                except IndexError:
                    ndxBegin = len(DeltaTL);
        
        OneMoreLoop = 0;
        if np.min(WaveUnique) == 0:
          MAX_ABS_TIMELAG = MAX_ABS_TIMELAG*0.75;
          OneMoreLoop = 1;
        iter_macro = iter_macro +1

    return Wave, WaveUnique, WaveSize, WaveTime, ExpectedTrans
    

def ReadPixelData(min_times):
    nTrans = np.zeros(len(min_times))
    pixelID = []
    
    for pixel in range(0, len(min_times)):
        nTrans[pixel] = len(min_times[pixel][1])
        if len(min_times[pixel][1])>0:
            pixelID.append(pixel)
            
    nPixel=len(pixelID);
    
    UpTrans = [];
    PixelLabel = [];

    for idx in pixelID:
       UpTrans.extend(min_times[idx][1].rescale(pq.s));
       #UpTrans.extend(min_times[idx][1].rescale(pq.ms));

       PixelLabel.extend(idx*np.ones(len(min_times[idx][1])));

    UpTrans = np.array(UpTrans)
    PixelLabel = np.array(PixelLabel)
    
    Sorted_Idx = np.argsort(UpTrans)
    UpTrans = UpTrans[Sorted_Idx]
    PixelLabel = PixelLabel[Sorted_Idx]
    
    return(nPixel, pixelID, nTrans, UpTrans, PixelLabel)
    
def NewDataFormat(events, DIM_X, DIM_Y):

    min_times = []
    print('X', DIM_X)
    for y in range(0, DIM_Y):
        temp2 = np.where(events.array_annotations['y_coords'] == y)[0]
        if len(temp2)>0:
            for x in range(0, DIM_X):
                temp1 = np.where(events.array_annotations['x_coords'] == x)[0]
                temp = [np.intersect1d(temp1,temp2)]
                idx = temp[0]
                time = events.times[idx]
                pixel = y*DIM_Y + x
                point = [pixel]
                point.append(time)
                min_times.append(point)
        else:
            for x in range(0, DIM_X):
                pixel = y*DIM_Y + x
                point = [pixel]
                point.append([])
                min_times.append(point)
    return(min_times)

# ======================================================================================#

# LOAD input data
if __name__ == '__main__':
    
    CLI = argparse.ArgumentParser(description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                        help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                        help="path of output file")
    
    CLI.add_argument("--DIM_X", nargs='?', type=int, default=50,
                        help="number of channel along x dimension")
    CLI.add_argument("--DIM_Y", nargs='?', type=int, default=50,
                        help="fnumber of channel along x dimensio")
    CLI.add_argument("--Max_Abs_Timelag", nargs='?', type=float, default=0.8,
                        help="Maximum reasonable time lag between electrodes (pixels)")
    CLI.add_argument("--Acceptable_rejection_rate", nargs='?', type=float, default=0.1,
                        help=" ")
    CLI.add_argument("--MIN_CH_NUM", nargs='?', type=float, default=300,
                        help=" ")
  
    
    CLI.add_argument("--metric", nargs='?', type=str, default='euclidean',
                        help="parameter for sklearn.cluster.DBSCAN")
    CLI.add_argument("--time_space_ratio", nargs='?', type=float, default=1,
                        help="factor to apply to time values")
    CLI.add_argument("--neighbour_distance", nargs='?', type=float, default=30,
                        help="eps parameter in sklearn.cluster.DBSCAN")
    CLI.add_argument("--min_samples", nargs='?', type=int, default=10,
                        help="minimum number of trigger times to form a wavefront")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

    
    # Dataset Loading
    min_times = NewDataFormat(evts, args.DIM_X, args.DIM_Y)
    nPixel, pixelID, nTrans, UpTrans, PixelLabel = ReadPixelData(min_times)
    ChLabel = PixelLabel;

    Wave, WaveUnique, WaveSize, WaveTime, ExpectedTrans = Optimal_MAX_ABS_TIMELAG(nTrans, UpTrans, ChLabel, args.Max_Abs_Timelag)
    
    Stored = []
    Label = []
    WaveUnique_Save = []
    WaveTime_Save = []
    WaveSize_Save = []

    for i in range(0, len(Wave)):
        Stored.extend(list(Wave[i]['ndx']))
        Label.extend([i for elem in range(0, len(Wave[i]['ndx']))])
        WaveUnique_Save.extend([WaveUnique[i] for elem in range(0, len(Wave[i]['ndx']))])
        WaveSize_Save.extend([WaveSize[i] for elem in range(0, len(Wave[i]['ndx']))])
        WaveTime_Save.extend([WaveTime[i] for elem in range(0, len(Wave[i]['ndx']))])

    Stored = {'Ops': Stored*pq.s}
    Waves_Inter = neo.Event(times=Stored['Ops'],
                    labels = Label,
                    name='Wavefronts',
                    array_annotations={'channels':WaveUnique_Save,
                                       'x_coords':WaveSize_Save,
                                       'y_coords':WaveTime_Save},
                    description='Transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    ExpectedTrans=ExpectedTrans)
    
    
    
    remove_annotations(Waves_Inter, del_keys=['nix_name', 'neo_name'])
    Waves_Inter.annotations.update(Waves_Inter.annotations)
    block.segments[0].events.append(Waves_Inter)
    write_neo(args.output, block)

 


