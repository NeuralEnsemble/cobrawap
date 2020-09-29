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


def MAX_IWI_Search(UpTrans,nCh,ChannelSet,ChLabel, Wave, WaveSize, WaveTime, ExpectedTrans, ACCEPTABLE_REJECTION_RATE):

    ## Wave Hunt -- step 1: Compute the Time Lags, Find Unique Waves --> WaveCollection1
    X=list(range(0,nCh));
    dX = X[1] - X[0]
    Size = np.zeros(nCh)
    for i in X:
        Size[i] = len(np.where(np.array(WaveSize[:])==i)[0]);
    WaveColl = [{'nWaves': len(WaveSize), 'NumberSize': Size}];
    IWI = np.diff(WaveTime);          # IWI = Inter-Wave-Interval...


    ##Recollect small-waves in full waves (involving a wider area).
    
    MAX_IWI = np.max(IWI)*0.5
    medianIWI = np.median(IWI[np.where(IWI>=0.250)]); # estimate TAKING INTO ACCOUNT the 4Hz LIMIT
    print('MaxIWI', MAX_IWI)
    
    iter = 0
    OneMoreLoop = 1;
    while OneMoreLoop:
        WnW = IWI<=MAX_IWI;
        ndxBegin = np.where(WnW==1)[0][0];
        nw = -1;
        FullWave = []
        FullWaveUnique = []
        FullWaveUniqueSize = []
        FullWaveSize = []
        FullWaveTime = []
        iter = 0
        while ndxBegin < len(IWI):
        
            try:
              ndxEnd = np.where(WnW[ndxBegin:len(WnW)]==0)[0][0]+ndxBegin; # isolated transitions are EXCLUDED
            except IndexError:
              ndxEnd = len(WaveTime)-1;
            
            
            nw = nw + 1;
            
            
            ndx = list(range(ndxBegin,ndxEnd+1))
            
            Full_ndx = []
            for elem in ndx:
                Full_ndx.extend(Wave[elem]['ndx'])
            Full_ndx = np.int64(Full_ndx)
            
            FullWave.append({'ndx': Full_ndx});
            FullWaveUnique.append(len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])));
            FullWaveSize.append(len(Full_ndx));
            FullWaveTime.append(np.mean(UpTrans[Full_ndx]));
            FullWaveUniqueSize.append(len(np.unique(ChLabel[Full_ndx])));

            
            if ndxEnd == len(WaveTime)-1:
                ndxBegin = len(IWI);
                
            else:
                try:
                    ndxBegin = np.where(WnW[ndxEnd:len(WnW)]==1)[0][0]+ndxEnd;
                except IndexError:
                    ndxBegin = len(DeltaTL);
                

                for j in range(ndxEnd+1,ndxBegin):
                    nw = nw + 1;
                    ndx = [j]
                    Full_ndx = []
                    for elem in ndx:
                        Full_ndx.extend(Wave[elem]['ndx'])
                    Full_ndx = np.int64(Full_ndx)
                    print('Full ndx', Full_ndx)
                    FullWave.append({'ndx': Full_ndx});
                    FullWaveUnique.append(len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])));
                    FullWaveSize.append(len(Full_ndx));
                    FullWaveTime.append(np.mean(UpTrans[Full_ndx]));
                    FullWaveUniqueSize.append(len(np.unique(ChLabel[Full_ndx])));
                
            iter = iter + 1
            
        BadWavesNum = len(FullWaveUnique) - len(np.where(FullWaveUnique)[0]);
        print('Max wave size: %d, Bad waves: %d (%d), Good waves: %d\n' %
               (max(FullWaveSize),BadWavesNum,BadWavesNum/len(FullWaveUnique)*100,len(FullWaveUnique)-BadWavesNum));
        print('Num. waves: %d, max. non-unique ch.: %d\n' %
               (len(FullWaveUnique),np.max(np.array(FullWaveSize)-np.array(FullWaveUniqueSize))));

        OneMoreLoop = 0;
        if len(FullWaveUnique) <= ExpectedTrans: # If not we have an artifactual amplification of small waves...
            if float(BadWavesNum)/len(FullWaveUnique) > ACCEPTABLE_REJECTION_RATE:
                if np.min(FullWaveUnique) == 0: # at lest a Wave non-unique
                    print('MaxIWI too large: %f -> %f\n'% (MAX_IWI,MAX_IWI*0.75));
                    MAX_IWI = MAX_IWI*0.75;
                    OneMoreLoop = 1;
                else: # only unique waves
                    if np.max(WaveSize) < len(ChannelSet): # at least one wave MUST BE GLOBAL (i.e. involving the full set of electrodes)
                        printf('MaxIWI too small: %f -> %f\n'% (MAX_IWI,MAX_IWI*1.25));
                        MAX_IWI = MAX_IWI*1.25;
                        OneMoreLoop = 1;
        iter = iter +1

    
    print('\nExpected waves     : %d\n' % round(ExpectedTrans));
    print('Reconstructed waves: %d\n' % len(FullWaveUnique));
    totFullWaveTrans=np.sum(FullWaveSize);
    
    return(FullWave, FullWaveUnique, FullWaveSize, FullWaveTime, FullWaveUniqueSize, MAX_IWI)


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
    
def ArrayMaskCreation(height, width):
    #LOCALITY criteria
    ArrayMask = [];
    h = height;
    w = width;

    for i in range(0,h*w):
        
        neigh = [];
        
        #right
        if (i+1) % w != 1:
            neigh.append(i+1);
        if (i+w+1)%w != 1 & (i+w+1)<(w*h+1):
            neigh.append(i+w+1);
        if (i-w+1)%w != 1 & (i-w+1)>1:
            neigh.append(i-w+1);
    
        #left
        if (i-1)%w != 0:
            neigh.append(i-1);
        if ((i-w-1)%w) != 0 & (i-w-1)>0:
            neigh.append(i-w-1)
        if ((i+w-1)%w) != 0 & (i+w-1)<(w*h):
            neigh.append(i+w-1);

        #spigoli
        if ((i+w) < (w*h+1)):
            neigh.append(i+w)
        if ((i-w) > 0):
            neigh.append(i-w);
            
        ArrayMask.append({'neigh': neigh});
        

    return(ArrayMask)

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
    print()
    block = load_neo(args.metric)
    asig = block.segments[0].analogsignals[0]
    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

    # Dataset Loading
    min_times = NewDataFormat(evts, args.DIM_X, args.DIM_Y)
    nPixel, pixelID, nTrans, UpTrans, PixelLabel = ReadPixelData(min_times)
    nCh = nPixel;
    RecordingSet = list(range(1,nCh));
    ChLabel = PixelLabel;
    ChannelSet = RecordingSet;
        
    # Previous block preprocessing
    block_bis = load_neo(args.data)
    asig = block_bis.segments[0].analogsignals[0]
    evts = [ev for ev in block_bis.segments[0].events if ev.name== 'Wavefronts']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")
    
    NTot = np.unique(np.int64(evts.labels))
    
    Wave = []
    WaveSize = []
    WaveTime = []
    WaveUnique = []
    for w in NTot:
        idx = np.where(np.int64(evts.labels) == w)[0]
        ndx = list(evts.times[idx].magnitude)
        Wave.append({'ndx': ndx})
        WaveSize.append(evts.array_annotations['x_coords'][int(idx[0])])
        WaveTime.append(evts.array_annotations['y_coords'][int(idx[0])])
        WaveUnique.append(evts.array_annotations['channels'][int(idx[0])])
            
    FullWave, FullWaveUnique, FullWaveSize, FullWaveTime, FullWaveUniqueSize, MAX_IWI = MAX_IWI_Search(UpTrans,nCh,ChannelSet,ChLabel, Wave, WaveSize, WaveTime, evts.annotations['ExpectedTrans'], args.Acceptable_rejection_rate)
    
    Stored = []
    Label = []
    FullWaveUnique_Save = []
    FullWaveTime_Save = []
    FullWaveSize_Save = []
    FullWaveUniqueSize_Save = []

    
    for i in range(0, len(FullWave)):
        Stored.extend(list(FullWave[i]['ndx']))
        Label.extend([i for elem in range(0, len(FullWave[i]['ndx']))])
        FullWaveUnique_Save.extend([FullWaveUnique[i] for elem in range(0, len(FullWave[i]['ndx']))])
        FullWaveSize_Save.extend([FullWaveSize[i] for elem in range(0, len(FullWave[i]['ndx']))])
        FullWaveTime_Save.extend([FullWaveTime[i] for elem in range(0, len(FullWave[i]['ndx']))])
        FullWaveUniqueSize_Save.extend([FullWaveUniqueSize[i] for elem in range(0, len(FullWave[i]['ndx']))])

    Stored = {'Ops': Stored*pq.s}
    Waves_Inter = neo.Event(times=Stored['Ops'],
                    labels=Label,
                    name='Wavefronts',
                    array_annotations={'channels':FullWaveUnique_Save,
                                       'x_coords':FullWaveSize_Save,
                                       'y_coords':FullWaveTime_Save,
                                       'ops':FullWaveUniqueSize_Save},
                    description='Transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    MAX_IWI=MAX_IWI)


    
    remove_annotations(Waves_Inter, del_keys=['nix_name', 'neo_name'])
    Waves_Inter.annotations.update(Waves_Inter.annotations)
    block.segments[0].events.append(Waves_Inter)
    write_neo(args.output, block)

    



