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


def WaveHuntTimes(nTrans, UpTrans, nPixel,pixelID,nCh,RecordingSet,ChannelSet,ChLabel, MAX_ABS_TIMELAG, ACCEPTABLE_REJECTION_RATE, meanUP, THR1, THR2, ArrayMask, DataDir,SPATIAL_SCALE, DIM_X, DIM_Y,     MIN_CH_NUM):

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


    #### ! FINO A QUI OK! ###
    print('LenWave', len(Wave))
    print('Wave 0', Wave[0])
    print('################################################################################')
    
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
            
            print('ndxEnd', ndxEnd)
            
            nw = nw + 1;
            
            print('nw', nw)
            
            ndx = list(range(ndxBegin,ndxEnd+1))
            
            Full_ndx = []
            for elem in ndx:
                Full_ndx.extend(Wave[elem]['ndx'])
            print('Full_ndx', len(Full_ndx))
            
            
            FullWave.append({'ndx': Full_ndx});
            print('le FullWave nw', len(FullWave[nw]['ndx']))
            
            FullWaveUnique.append(len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])));
            print('FullWaveUnique', FullWaveUnique[nw])
            
            FullWaveSize.append(len(Full_ndx));
            print('FullWaveSize', FullWaveSize[nw])
            
            FullWaveTime.append(np.mean(UpTrans[Full_ndx]));
            print('FullWaveTime', FullWaveTime[nw])
            
            FullWaveUniqueSize.append(len(np.unique(ChLabel[Full_ndx])));
            print('FullWaveUniqueSize', FullWaveUniqueSize[nw])

            
            if ndxEnd == len(WaveTime)-1:
                ndxBegin = len(IWI);
                
                print('ndxBegin2', ndxBegin)
            else:
                try:
                    ndxBegin = np.where(WnW[ndxEnd:len(WnW)]==1)[0][0]+ndxEnd;
                except IndexError:
                    ndxBegin = len(DeltaTL);
                
                print('ndxBegin2', ndxBegin)

                for j in range(ndxEnd+1,ndxBegin):
                    nw = nw + 1;
                    ndx = [j]
                    Full_ndx = []
                    for elem in ndx:
                        Full_ndx.extend(Wave[elem]['ndx'])

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
    

    ###################################################################################################
    ## Remove from waves nonunique channels...

    FullWaveBackup = FullWave;
    TOTNonUnique=len(FullWaveUnique) - len(np.where(FullWaveUnique)[0]);
    
    print('FullWaveUniqueSize',FullWaveUniqueSize)
    print('FullWaveSize',FullWaveSize)
    print('Diff', np.max(np.array(FullWaveSize)-np.array(FullWaveUniqueSize)))


    alert = np.zeros(6)
    
    print('\tWaves to clean     : %d\n' % (TOTNonUnique));
    print('\t(Max. non-unique ch.: %d)\n'% (np.max(np.array(FullWaveSize)-np.array(FullWaveUniqueSize))));
    
    #### FINO A QUI OK ###
    
    listOfnw=[]; # List of "critical" waves: non-unique but possibly hiding multiple unique waves
    nAdded = 0; # total number of new waves added to the collection
    nNew=[];
    nSegments=[]; # (how many segments non-unique waves are made of)
    nSize=[]; # (size of the segments)

    nw=0;
    
    NonUnique=0; # COUNTER for NonUnique Waves
    
    iter_macro = 0
    
    while nw<len(FullWaveUnique): # for each wave
        
        print('---------------------------------- ITER MACRO:', iter_macro)
        print('---------------------------------- nw:', nw)

        print('----------------------------------NUOVO CICLO -------------------------------------------------')
        NewFullWave=[];NewFullWaveUnique=[];NewFullWaveSize=[];NewFullWaveTime=[];seg=[]; # empty arrays
        nCHS=[0, 0];


        if FullWaveUnique[nw] == 0:
            NonUnique=NonUnique+1;
            print('len Full Wave', len(FullWave[nw]['ndx']))
            
            chs = ChLabel[FullWave[nw]['ndx']];
            print('len chs', len(chs))
            
            nCHS[0]=len(chs);
            wndx = list(range(0,len(chs)));
            nhc, bins = np.histogram(chs, bins=list(range(0,len(RecordingSet)+1)))
            print('Rec', len(RecordingSet))
            print('nch', np.array(nhc))
            # --- (1) First CHANNEL CLEANING: check the TIME DISTANCE BETWEEN REPETITIONS
            rep=np.where(nhc > 1)[0];
            print('len rep', len(rep))
            
            k=0;
            
            while k<len(rep):
                i=rep[k];
                temp = []
                temp = np.where(chs == i)[0]
                timeStamp = []
                idx = []
                for elem in temp:
                    timeStamp.append(UpTrans[FullWave[nw]['ndx'][elem]]);
                    idx.append(FullWave[nw]['ndx'][elem]);

                
                try:
                    if np.diff(timeStamp)<0.125:
                        # open a time-window around each occurrence of i and check
                        # how many members of the clan are in the time-window
                        print('Channel %d has close repetitions\n'% (i));
                        nClan=[];
                        for j in range(0, nhc[i]):
                            window=[timeStamp[j]-0.125, timeStamp[j]+0.125];
                            temp = UpTrans[np.where(FullWave[nw]['ndx'] > window[0] and FullWave[nw]['ndx'] < window[1])[0]]
                            inWindow=FullWave[nw]['ndx'][temp];
                            
                            inWindow=np.setdiff1d(inWindow,idx[j]);
                            nClan.append(len(np.intersect1d(ChLabel[inWindow], arrayMask[i]['neigh'])));
                
                        if np.where(nClan==0): # to be more generic, consider the case min(nClan)
                            [FullWave[nw]['ndx'],a]=np.setdiff1d(FullWave[nw]['ndx'],idx[np.where(nClan==0)]);
                            b=np.setdiff1d(wndx,a);
                            chs[b]=[];
                            print('%d repetition(s) deleted\n'% (len(b)));
                            nch=np.histogram(chs,list(range(0,len(RecordingSet))));
                            if(nch[i]>1):
                                print('...Now, re-do channel %d\n' % (i));
                            else:
                                k=k+1;
                          
                        else:
                            print('CloseRepProblem for channel i = %d NOT SOLVED\n' % (i));
                            k=k+1;
                    else:
                        k=k+1;
                except ValueError:
                #except np.core._internal.AxisError:
                    k=k+1;
                
            nCHS[1]=len(chs); # update the number of transitions (to keep count of removed transitions)
            
            #Check the time difference between transitions
            delta=np.diff(UpTrans[FullWave[nw]['ndx']]);
            mD=np.mean(delta);
            stdD=np.std(delta);
            # Look for CANDIDATE SEGMENTS
            THR=mD+3*stdD;
            B = []
            for elem in np.where(delta>THR)[0]:
                B.append(elem); # larger sepration between transitions
                                  # --> possible BREAK of the non-unique wave
            
            nSize = []
            nSegments = []
            
            if B: # --- IDENTIFY SEGMENTS
                # SEGMENTS i.e. candidate NEW waves
                try:
                    segments=len(B)+1;
                except TypeError:
                    segments = 2;
                    
                istart=0;
                i=0;
                seg = []
                
                while i < len(B):
                    seg.append({'ndx': list(range(istart,B[i]+1)),
                                'chs': chs[list(range(istart,B[i]+1))]
                               });
                    istart=B[i]+1;
                    i=i+1;
            
                seg.append({'ndx': list(range(istart,len(chs))),
                            'chs': chs[list(range(istart,len(chs)))]})
                
                # --- CLEAN SEGMENTS---
                delSeg=[];
                for i in range(0,segments): # CHECK if SEGMENTS have to be cleaned

                    # 1) non-unique segments
                    if len(np.unique(seg[i]['chs'])!=len(seg[i]['chs'])): # the subset of transition is NOT unique
                        # --> 'CLEAN', i.e. scan the transition sequence and
                        # find repetition of channels
                        print('alert[0] nw=%d, seg=%d\n' % (nw,i));
                        alert[0]=alert[0]+1;
                        delCh=[];
                        
                        for j in range(0, len(seg[i]['chs'])):
                            listNdx=[]
                            for elem in np.where(seg[i]['chs']==seg[i]['chs'][j])[0]:
                                listNdx.append(elem);
                            
                            if(len(listNdx)!=1):
                                t0Clan=0;
                                nClan=0;
                                for k in ArrayMask[int(seg[i]['chs'][j])]['neigh']:
                                    tClan = []
                                    for elem in np.where(seg[i]['chs']==k)[0]:
                                        tClan.append(UpTrans[FullWave[nw]['ndx'][seg[i]['ndx'][elem]]]);
                                    if tClan:
                                        nClan=nClan+1;
                                        t0Clan=t0Clan+np.mean(tClan);
                                
                                if nClan > 0:
                                    t0Clan= float(t0Clan)/nClan;
                                    tCh = []
                                    for elem in np.where(seg[i]['chs']==seg[i]['chs'][j])[0]:
                                        tCh.append(UpTrans[FullWave[nw]['ndx'][seg[i]['ndx'][elem]]]);
                                    
                                    t0Ch= np.min(abs(np.array(tCh)-np.array(t0Clan)));
                                    index = np.argmin(abs(np.array(tCh)-np.array(t0Clan)));
                                    delCh.append(np.setdiff1d(np.array(listNdx),np.array(listNdx[index])));

                        '''
                        for elem in np.unique(delCh):
                            seg[i]['chs'] = np.delete(seg[i]['chs'], elem)
                            seg[i]['ndx']=np.delete(seg[i]['ndx'], elem); #PIPPo
                        '''
                        if len(delCh):
                            
                            temp = []
                            for elem in delCh:
                                temp.extend(elem)
                            delCh = temp

                            if len(np.unique(delCh)):
                                seg[i]['chs'] = np.delete(seg[i]['chs'], np.unique(delCh))
                                seg[i]['ndx']=np.delete(seg[i]['ndx'], np.unique(delCh)); #PIPPo

                    # 2) channels non-LocallyConnected (see arrayMask)
                    if len(seg[i]['chs'])<=5: # 5 is min(len(arrayMask{:}{2})
                        delList=[];
                        for j in range(0,len(seg[i]['chs'])):
                            k=seg[i]['chs'][j];
                            if not (np.intersect1d(ArrayMask[int(k)]['neigh'], np.setdiff1d(seg[i]['chs'],k))).size:
                                delList.append(j);
                                
                        if delList:
                            print('alert[1] nw=%d, seg=%d\n' % (nw,i));
                            alert[1]=alert[1]+1;
                            #seg[i]['chs'][delList]=[];#PIPPo
                            #seg[i]['ndx'][delList]=[];#PIPPo
                            seg[i]['chs']= np.delete( seg[i]['chs'],delList) 
                            seg[i]['ndx']= np.delete( seg[i]['ndx'],delList)


                    # PREPARE TO REMOVE EMPTY SEGMENTS
                    if len(seg[i]['ndx'])==0:
                        print('alert[2] nw=%d, seg=%d\n' % (nw,i));
                        alert[2]=alert[2]+1;
                        delSeg.append(i);
                        
                # REMOVE EMPTY SEGMENTS
                
                if delSeg:
                   seg=np.delete(seg,delSeg);
                   
                
                segments=len(seg); # update the value in 'segments' = number of segments

                # coalescence of segments if no repetitions with adjacent(s) one(s)
                # N.B. a "small" intersection is admitted
                i=0;
                
                while i<(segments-1):
                    if len(np.intersect1d(seg[i]['chs'],seg[i+1]['chs'])) <= np.floor(1/4*np.min([len(seg[i]['chs']),len(seg[i+1]['chs'])])):
                        # CANDIDATE SEGMENTS for COALESCENCE
                        # check also if distance between segments'border is smaller than 250ms = 1/4Hz
                        
                        distance = UpTrans[FullWave[nw]['ndx'][seg[i+1]['ndx'][0]]] - UpTrans[FullWave[nw]['ndx'][seg[i]['ndx'][len(seg[i]['ndx'])-1]]];
                        
                        if distance>=0.250:
                            # FREQUENCY ALERT: distance compatible with SWA, the two segments should be kept separated
                            print('>>> alert[3] (Frequency Alert)\n nw=%d, seg=%d AND seg=%d\n' % (nw,i,i+1) );
                            alert[3]=alert[3]+1;
                            print('TOT Transitions: %d\n'% (len(seg[i]['chs'])+len(seg[i+1]['chs'])));
                            print('len(intesect): %d\n'% (len(np.intersect1d(seg[i]['chs'], (seg[i+1]['chs'])))));
                            print('threshold: %d\n'% (np.floor(1/4*np.min([len(seg[i]['chs']), (len(seg[i+1]['chs']))]))));
                            i=i+1; #increment the pointer only if no coalescence is made
                        else:
                            # COALESCENCE
                            # The two segments are close enough that can be merged into a single wave
                            # (consider them separated waves would mean the SWA frequency is larger than 4Hz)
                            
                            print('alert[4] nw=%d, seg=%d AND seg=%d\n' % (nw,i,i+1));
                            alert[4]=alert[4]+1;
                            print('TOT Transitions: %d\n' % (len(seg[i]['chs'])+len(seg[i+1]['chs'])));
                            print('len(intesect): %d\n' % (len(np.intersect1d(seg[i]['chs'],seg[i+1]['chs']))));
                            print('threshold: %d\n'% (np.floor(1/4*min(len(seg[i]['chs']),len(seg[i+1]['chs'])))));

                            # COALESCENCE of consecutive SEGMENTS
                            
                            mergedCHS=list(seg[i]['chs'])
                            mergedCHS.extend(seg[i+1]['chs']);
                            mergedNDX=list(seg[i]['ndx']);
                            mergedNDX.extend(seg[i+1]['ndx'])
                            
                            # CHECK for REPETITIONS (and treat them as usual...
                            # looking at the meanTime in the Clan)
                            delCh=[];
                            for j in range(0,len(mergedCHS)):
                                listNdx = []
                                for elem in np.where(mergedCHS==mergedCHS[j])[0]:
                                    listNdx.append(elem);
                                
                                if len(listNdx)!=1:
                                    t0Clan=0;
                                    nClan=0;
                                    for k in ArrayMask[int(mergedCHS[j])]['neigh']:
                                        tClan = []
                                        
                                        for elem in np.where(mergedCHS==k)[0]:
                                            tClan.append(UpTrans[FullWave[nw]['ndx'][mergedNDX[elem]]])

                                        
                                        if tClan:
                                            nClan=nClan+1;
                                            t0Clan=t0Clan+tClan;
                                    try:
                                        t0Clan=np.float(t0Clan)/nClan;
                                    except ZeroDivisionError:
                                        t0Clan = 0
                                    tCh = []
                                    for elem in np.where(mergedCHS==mergedCHS[j])[0]:
                                        tCh.append(UpTrans[FullWave[nw]['ndx'][mergedNDX[elem]]]);
                                    t0Ch= np.min(abs(np.array(tCh)-t0Clan));
                                    index= np.argmin(abs(np.array(tCh)-t0Clan));
                                    delCh.append(np.setdiff1d(np.array(listNdx),np.array(listNdx[index])));

                            if len(delCh):
                                
                                temp = []
                                for elem in delCh:
                                    temp.extend(elem)
                                delCh = temp

                                if len(np.unique(delCh))>0:
                                    mergedCHS = np.delete(mergedCHS, np.unique(delCh));
                                    mergedNDX = np.delete(mergedNDX, np.unique(delCh));
                            
                            seg[i]['chs']=mergedCHS; # COALESCENCE
                            seg[i]['ndx']=mergedNDX; # COALESCENCE
                            seg=np.delete(seg, i+1); # coalesced segments are at index i, segment at index i+1 is REMOVED
                            segments=segments-1;
                            
                    else: # consecutive segments intersect too much...
                        i=i+1; #increment the pointer only if no coalescence is made


                                
                if segments!=len(seg):
                    print('ERROR - Number of Segments');
                

                # $$$$$ N.B. the number of segments has to be updated
                NewFullWave = []
                NewFullWaveUnique = []
                NewFullWaveSize = []
                NewFullWaveTime = []

                
                for i in range(0,segments):
                    ndx = []
                    for elem in seg[i]['ndx']:
                        ndx.append(FullWave[nw]['ndx'][elem])
                    NewFullWave.append({'ndx': ndx});
                    NewFullWaveUnique.append(1); # update the logical value (...we are "cleaning" the waves)
                    NewFullWaveSize.append(len(NewFullWave[i]['ndx']));
                    NewFullWaveTime.append(np.mean(UpTrans[NewFullWave[i]['ndx']]));
                    nSize.append(NewFullWaveSize[i]);

                nSegments.append(segments);
                

            else: # NO SEGMENTS identified -->
                 # repeated chiannels are due to noise (and not to the presence of more than one wave)
                 # CLEAN the wave, i.e. keep only the first channel occurrance
                print('alert[5] nw=%d\n', nw);
                alert[5]=alert[5]+1;
                delCh=[];
                for j in range(0,len(chs)):
                    listNdx= np.where(chs==chs[j])[0];
                    if(len(listNdx)!=1):
                        # Keep the occurrance which is the closest to the other occurances in the "clan"
                        t0Clan=0;
                        nClan=0;
                        tClan = []
                        for k in ArrayMask[int(chs[j])]['neigh']:
                            for elem in np.where(chs==k)[0]:
                                tClan.append(UpTrans[FullWave[nw]['ndx'][elem]]);
                            if tClan:
                                try:
                                    nClan=nClan+len(tClan); # take into account the case the CLAN has repetions
                                    t0Clan=t0Clan+sum(tClan); # take into account the case the CLAN has repetions
                                except TypeError:
                                    nClan=nClan +1
                                    t0Clan=t0Clan+tClan
                                    
                        t0Clan=float(t0Clan)/nClan;
                        tCh = []
                        for elem in np.where(chs==chs[j])[0]:
                            tCh.append(UpTrans[FullWave[nw]['ndx'][elem]]);
                        t0Ch=np.min(abs(np.array(tCh)-t0Clan));
                        index=np.argmin(abs(np.array(tCh)-t0Clan));

                        delCh.append(np.setdiff1d(listNdx,listNdx[index]));
                        #print('delch:', delCh)

                #print(chs)
                ''' CAMBIO QUI
                for elem in np.unique(delCh):
                    FullWave[nw]['ndx'][elem]=0;
                    chs[elem]=0;
                '''
                
                if len(np.unique(delCh)):
                    FullWave[nw]['ndx'] = np.delete(FullWave[nw]['ndx'], np.unique(delCh))
                    chs = np.delete(chs, np.unique(delCh))

                # wave is "cleaned"; store and plot the updated wave
                ndx = FullWave[nw]['ndx']
                NewFullWave = {'ndx': ndx};
                NewFullWaveUnique = 1; # update the logical value (...we are "cleaning" the waves)
                NewFullWaveSize = len(ndx);
                NewFullWaveTime = np.mean(UpTrans[ndx]);
                

                nSize.append(NewFullWaveSize);
                nSegments.append(1);

            # --- REPLACE CurrentWave with NewWave(s)
            # [its segments or its 'cleaned' version]
            
            if nw != 0:
                Pre = FullWave[:].copy()
                FullWave=Pre[0:nw].copy()
                FullWave.extend(NewFullWave)
                FullWave.extend(Pre[nw+1:]);#end

                Pre = FullWaveUnique[:].copy()
                FullWaveUnique = Pre[0:nw].copy()
                FullWaveUnique.extend(NewFullWaveUnique)
                FullWaveUnique.extend(Pre[nw+1:]); #end

                Pre = FullWaveSize[:].copy()
                FullWaveSize = Pre[0:nw].copy()
                FullWaveSize.extend(NewFullWaveSize)
                FullWaveSize.extend(Pre[nw+1:]); #end
                 
                Pre = FullWaveTime[:].copy()
                FullWaveTime = Pre[0:nw].copy()
                FullWaveTime.extend(NewFullWaveTime)
                FullWaveTime.extend(Pre[nw+1:]);#end
                
            else:
                
                Pre = FullWave[:].copy()
                FullWave = NewFullWave.copy()
                FullWave.extend(Pre[nw+1:]);#end
                
                Pre = FullWaveUnique[:].copy()
                FullWaveUnique = NewFullWaveUnique.copy()
                FullWaveUnique.extend(Pre[nw+1:]); #end
                
                Pre = FullWaveSize[:].copy()
                FullWaveSize = NewFullWaveSize.copy()
                FullWaveSize.extend(Pre[nw+1:]); #end
                
                Pre = FullWaveTime[:].copy()
                FullWaveTime = NewFullWaveTime.copy()
                FullWaveTime.extend(Pre[nw+1:]);#end
                 

            # --- INCREMENT the pointer
            if len(NewFullWave)>0: # SEGMENTS ARE NEW WAVES
                nAdded= nAdded + len(NewFullWave)-1;
                nw = nw+len(NewFullWave); # increment (point at the next wave)
            else: # no segments identified, the current wave is a New Wave, because it has been cleaned
                nw=nw+1; # increment (point at the next wave)                      

        else:
            nw=nw+1; # increment (point at the next wave) [current wave is already unique]

             
        if NewFullWave:
            nNew.append(len(NewFullWave));
     
        iter_macro = iter_macro +1
        

    print('alert = [%d,%d,%d,%d,%d,%d]\n', alert);

    ## Remove from waves nonunique channels...


    #print('\tWaves to clean     : %d\n' % (len(np.where(FullWaveUnique==0)[0])));
    #print('\t(Max. non-unique ch.: %d)\n'%  (np.max(np.array(FullWaveSize)-np.array(FullWaveUniqueSize))));
  
    # ATTENZIONE! Qui non ho inserito il pezzetto %% Remove from waves nonunique channels... if exist('OLDv','var')
    # dal momento che non sembra essere mai chiamato!

    # Wave Hunt -- step 2: Coalescence of 'Short' Waves, Rejection of 'Small' Waves --> WaveCollection2

    X=list(range(0,nCh));
    dX = X[1] - X[0]
    Size = np.zeros(nCh)
    for i in X:
        Size[i] = len(np.where(FullWaveSize==i));
    WaveColl.append({'nWaves': len(FullWaveSize), 'NumberSize': Size});
    value=sum(WaveColl[1]['NumberSize'][list(range(int(math.ceil(nCh/2)),nCh))])/WaveColl[1]['nWaves'];

    
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
        TLMs = UpTrans[Wave[k]['ndx']];
        CLs = ChLabel[Wave[k]['ndx']];
        
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
        WaveDuration.append(UpTrans[FullWave[i]['ndx'][len(FullWave[i]['ndx'])-1]]-UpTrans[FullWave[i]['ndx'][0]]);
    

    # Create an array with the beginning time of each wave
    # Saves the array in a 'BeginTime.txt' file in the path folder
    BeginTime = [];
    EndTime = [];
    BeginTimeFile = DataDir + 'BeginTime.txt'
    EndTimeFile = DataDir + 'EndTime.txt';
    Waves = []

    Times = []
    Label = []
    Pixels = []
    for i in range(0,len(Wave)):
        
        Times.extend(UpTrans[Wave[i]['ndx']])
        Label.extend(np.ones([len(UpTrans[Wave[i]['ndx']])])*i)
        Pixels.extend(ChLabel[Wave[i]['ndx']])
        
        BeginTime.append(UpTrans[Wave[i]['ndx'][0]]);
        EndTime.append(UpTrans[Wave[i]['ndx'][len(Wave[i]['ndx'])-1]]);

    BeginTime = BeginTime*pq.s
    EndTime = EndTime*pq.s
    np.savetxt(BeginTimeFile, BeginTime.rescale(pq.s))
    np.savetxt(EndTimeFile, EndTime.rescale(pq.s))

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
    CLI.add_argument("--meanUP", nargs='?', type=float, default=0.2,
                        help=" ")
    CLI.add_argument("--ThR1", nargs='?', type=float, default=0.04,
                        help=" ")
    CLI.add_argument("--ThR2", nargs='?', type=float, default=0.250,
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
    print('asig', np.shape(asig))
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

    
    #np.savetxt('min_times_origin.txt', evts.times)
    
    print(evts.times)
    #print((evts.times).rescale(pq.ms))

    #(evts.times).rescale(pq.ms)

    #print('SPATIAL SCALE', evts.spatial_scale)
    
    print('Inizio...')
    DataDir = ''
    
    # Dataset Loading
    min_times = NewDataFormat(evts, args.DIM_X, args.DIM_Y)

    np.savetxt('min_times.txt', min_times,  fmt='%s')
    io.savemat('min_times_mat.mat', {'times': min_times})
    
    nPixel, pixelID, nTrans, UpTrans, PixelLabel = ReadPixelData(min_times)

    nCh = nPixel;
    RecordingSet = list(range(1,nCh));
    ChLabel = PixelLabel;
    ChannelSet = RecordingSet;
    ArrayMask = ArrayMaskCreation(args.DIM_X, args.DIM_Y)

    WaveHunt = WaveHuntTimes(nTrans, UpTrans, nPixel,pixelID,nCh,RecordingSet,ChannelSet,ChLabel, args.Max_Abs_Timelag, args.Acceptable_rejection_rate, args.meanUP, args.ThR1, args.ThR2, ArrayMask, DataDir, evts.annotations['spatial_scale'], args.DIM_X, args.DIM_Y, args.MIN_CH_NUM)
    
    block.segments[0].events.append(WaveHunt)
    write_neo(args.output, block)

    



