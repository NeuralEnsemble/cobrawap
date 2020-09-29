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

def WaveHuntTimes_Globality(UpTrans,nCh,RecordingSet,ChLabel,SPATIAL_SCALE,MIN_CH_NUM, FullWave, FullWaveUnique, FullWaveSize, FullWaveTime, DIM_X,DIM_Y):

    # Wave Hunt -- step 2: Coalescence of 'Short' Waves, Rejection of 'Small' Waves --> WaveCollection2
    WaveColl = []
    X=list(range(0,nCh));
    dX = X[1] - X[0]
    Size = np.zeros(nCh)
    for i in X:
        Size[i] = len(np.where(FullWaveSize==i));
    WaveColl.append({'nWaves': len(FullWaveSize), 'NumberSize': Size});
    value=sum(WaveColl[0]['NumberSize'][list(range(int(math.ceil(nCh/2)),nCh))])/WaveColl[0]['nWaves'];

    
    ## Find Maxima and Minima in the histogram of FullWaveSize

    hh, xx = np.histogram(FullWaveSize, bins=list(range(0,nCh)))

    for i in range(0,len(hh)-1):
        if abs(hh[i]-hh[i+1])==1:
            hh[i+1]=hh[i];

    RI=np.diff(hh);
    bMax=[];
    bMin=[];
    
    for i in range(0,len(RI)-2):
        if np.sign(RI[i])*np.sign(RI[i+1])!=1:
            if np.sign(RI[i])!=0:
                if np.sign(RI[i])>np.sign(RI[i+1]):
                    bMax.append(i+1);
                else:
                    bMin.append(i+1);
                
            else:
                if len(np.intersect1d(i,bMin))>0:
                    bMin.append(i+1);
                
                if len(np.intersect1d(i,bMax))>0:
                    bMin.append(i+1);


    if hh[len(hh)-1]>hh[len(hh)-2]:
        print('BIN %d is a MAXIMUM\n'%(len(hh)));
        bMax.append(i+1);


    # Remove small waves and those rejected...
        

    temp = [FullWaveUnique[i]==1 and FullWaveSize[i] >= MIN_CH_NUM  for i in range(0, len(FullWaveUnique))]
    ndx = np.where(np.array(temp))[0];
    RejectedWaves = len(FullWaveUnique) - len(ndx); # rejected beacuse too small
                                                    # (i.e. involving too few channels)
    Wave = []
    WaveUnique = []
    WaveSize = []
    WaveTime = []
    
    for idx in ndx:
        Wave.append(FullWave[idx]);
        WaveUnique.append(FullWaveUnique[idx]);
        WaveSize.append(FullWaveSize[idx]);
        WaveTime.append(FullWaveTime[idx]);

    print('\nMinimum number of channels: %d\n' % MIN_CH_NUM);
    print('Rejected waves: %d (too small, i.e. involve less than minimum number of channels)\n' % RejectedWaves);
    print('Accepted waves: %d\n' % len(Wave));
    
    WaveColl.append({'nWaves':len(Wave)});
    
    
    TimeLagMatrix = np.empty([len(Wave),len(RecordingSet)], dtype=int); # TLM initialized with NaN
    TimeLagMatrix[:].fill(np.nan)
    UpTransNdxMatrix = np.empty([len(Wave),len(RecordingSet)], dtype = int); # UpTransMatrix initialized with NaN
    UpTransNdxMatrix.fill(np.nan)
    
    for k in range(0,len(Wave)):
        PXs = [];
        TLMs = UpTrans[np.int64(Wave[k]['ndx'])];
        CLs = ChLabel[np.int64(Wave[k]['ndx'])];
        
        for i in range(0, len(CLs)):
        
            PXs.append(np.where(pixelID == CLs[i])[0]);


        #TimeLagMatrix[k, PXs] = np.array([TLMs]).T - np.mean(TLMs); # each wave is centered at the mean time
        #UpTransNdxMatrix[k,np.array(CLs, dtype=int)] = Wave[k]['ndx'];
    
    #TimeLagRange = [np.min(TimeLagMatrix[np.isnan(TimeLagMatrix)==0]), np.max(TimeLagMatrix[np.isnan(TimeLagMatrix)==0])]; # max duration of the# waves
    #TLMtoPlot = TimeLagMatrix;
    #TLMtoPlot[np.isnan(TLMtoPlot)==1] = -1; # '-1' replace 'NaN' in the plot
    
    
    # Duration of Waves

    WaveDuration=[];
    
    for i in range(0,len(FullWave)):
        WaveDuration.append(UpTrans[np.int64(FullWave[i]['ndx'][len(FullWave[i]['ndx'])-1])]-UpTrans[np.int64(FullWave[i]['ndx'][0])]);
    

    # Create an array with the beginning time of each wave
    # Saves the array in a 'BeginTime.txt' file in the path folder
    BeginTime = [];
    EndTime = [];

    Waves = []

    Times = []
    Label = []
    Pixels = []
    for i in range(0,len(Wave)):
        
        Times.extend(UpTrans[np.int64(Wave[i]['ndx'])])
        Label.extend(np.ones([len(UpTrans[np.int64(Wave[i]['ndx'])])])*i)
        Pixels.extend(ChLabel[np.int64(Wave[i]['ndx'])])
        
        BeginTime.append(UpTrans[np.int64(Wave[i]['ndx'][0])]);
        EndTime.append(UpTrans[np.int64(Wave[i]['ndx'][len(Wave[i]['ndx'])-1])]);

    BeginTime = BeginTime*pq.s
    EndTime = EndTime*pq.s


    Times = Times*pq.s
    waves = neo.Event(times=Times.rescale(pq.s),
                    labels=Label,
                    name='Wavefronts',
                    array_annotations={'channels':Pixels,
                                       'x_coords':[p % DIM_Y for p in Pixels],
                                       'y_coords':[np.floor(p/DIM_Y) for p in Pixels]},
                    description='Transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    spatial_scale=SPATIAL_SCALE)

    remove_annotations(waves, del_keys=['nix_name', 'neo_name'])
    waves.annotations.update(waves.annotations)
    return waves
    

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
    block = load_neo(args.metric)
    asig = block.segments[0].analogsignals[0]
    print(asig)
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
    spatial_scale = evts.annotations['spatial_scale']
    
    # Previous block preprocessing
    block_bis = load_neo(args.data)
    asig = block_bis.segments[0].analogsignals[0]
    print('ASIG', asig)
    evts = [ev for ev in block_bis.segments[0].events if ev.name== 'Wavefronts']
    print('evts', evts)

    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")
    print('Label', evts.labels)
    NTot = np.unique(np.int64(evts.labels))
    
    FullWave = []
    FullWaveSize = []
    FullWaveTime = []
    FullWaveUnique = []

    print('NTOT', NTot)
    for w in NTot:
        idx = np.where(np.int64(evts.labels) == w)[0]
        ndx = evts.times[idx].magnitude
        FullWave.append({'ndx': ndx})
        FullWaveSize.append(evts.array_annotations['x_coords'][idx[0]])
        FullWaveTime.append(evts.array_annotations['y_coords'][idx[0]])
        FullWaveUnique.append(evts.array_annotations['channels'][idx[0]])

    print('FullWave', FullWave)
    print('FullWave Size', FullWaveSize)
    print('FullWaveTime', FullWaveTime)
    print('FullWave Unique', FullWaveUnique)


    WaveHunt = WaveHuntTimes_Globality(UpTrans, nCh, RecordingSet, ChLabel, spatial_scale, args.MIN_CH_NUM, FullWave, FullWaveUnique, FullWaveSize, FullWaveTime, args.DIM_X, args.DIM_Y)

    block.segments[0].events.append(WaveHunt)
    write_neo(args.output, block)

    



