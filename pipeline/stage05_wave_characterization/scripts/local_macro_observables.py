import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import pandas as pd
from utils import load_neo, none_or_str, save_plot, AnalogSignal2ImageSequence
import math 


def calc_point_velocities(evts, DIM_X, DIM_Y):
    N_Waves = len(np.unique(evts.labels)) 
    WaveColl = np.ones([N_Waves, DIM_X, DIM_Y])*np.nan
    WaveColl[np.int32(np.float64(evts.labels)), np.int32(evts.array_annotations['x_coords']), np.int32(evts.array_annotations['y_coords'])] = evts.times
    WaveColl_2d = np.reshape(WaveColl, [np.shape(WaveColl)[0], np.shape(WaveColl)[1]*np.shape(WaveColl)[2]])
   
    #compute velocity
    Tx = np.diff(WaveColl, axis = 1)[:, 0:np.shape(WaveColl)[1]-1, 0:np.shape(WaveColl)[2]-1]
    Ty = np.diff(WaveColl, axis = 2)[:, 0:np.shape(WaveColl)[1]-1, 0:np.shape(WaveColl)[2]-1]

    Tx = np.reshape(Tx, [np.shape(Tx)[0], (np.shape(Tx)[1])*np.shape(Tx)[2]])
    Ty = np.reshape(Ty, [np.shape(Ty)[0], np.shape(Ty)[1]*(np.shape(Ty)[2])])
 
    vel =  np.sqrt((Tx**2 + Ty**2)/(2*evts.annotations['spatial_scale'])**2)*evts.times.units
    vel = 1./vel

    for v in vel:
        idx = np.where(np.isinf(v))[0]
        v[idx] = np.nan * evts.annotations['spatial_scale'].units/evts.times.units

    # compute direction
    direction = np.arctan2(Tx, Ty)

    
    #compute SO local IWI
    iwi = np.diff(WaveColl_2d, axis = 0)

    # create dictionary with results
    local_obs = {'velocity': vel, 'direction': direction, 'iwi': iwi}

    return(local_obs)

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    
    args = CLI.parse_args()
    block = load_neo(args.data)

    evts = [ev for ev in block.segments[0].events if ev.name == 'Wavefronts'][0] 
    
    # da miglirare!!!!
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[0]
    DIM_X = np.shape(imgseq)[1]
    DIM_Y = np.shape(imgseq)[2]
    local_obs_dict = calc_point_velocities(evts, DIM_X, DIM_Y)
    dict_1d = {}
    wave_id = []
    for i, w in enumerate(local_obs_dict['velocity']):
        wave_id.extend(np.ones(len(w))*i)

    if args.output_img is not None:
        
        plt.figure(figsize = (12, 3))

        plt.subplot(1,3,1)
        velocity = np.reshape(local_obs_dict['velocity'], np.shape(local_obs_dict['velocity'])[0]*np.shape(local_obs_dict['velocity'])[1])
        dict_1d['velocity'] = velocity
        velocity = velocity[np.where(~np.isnan(velocity))[0]]
        plt.hist(velocity.magnitude, color = [199./255., 0./255., 57./255.])
        plt.xlabel('local velocity ('+str(velocity.units)+')', fontsize =10.)
        
        plt.subplot(1,3,2, projection='polar')
        direction = np.reshape(local_obs_dict['direction'], np.shape(local_obs_dict['direction'])[0]*np.shape(local_obs_dict['direction'])[1])
        dict_1d['direction'] = direction
        direction = direction[np.where(~np.isnan(direction))[0]]
        plt.hist(direction, color = [23./255., 165./255., 137./255.])
        plt.xlabel('local direction', fontsize = 10.)
        
        plt.subplot(1,3,3)
        iwi = np.reshape(local_obs_dict['iwi'], np.shape(local_obs_dict['iwi'])[0]*np.shape(local_obs_dict['iwi'])[1])
        dict_1d['iwi'] = iwi
        iwi = iwi[np.where(~np.isnan(iwi))[0]]
        plt.hist(iwi, color = [230./255., 126./255., 34./255.])
        plt.xlabel('local iwi (Hz)', fontsize = 10.)
        
        plt.tight_layout()
        plt.savefig(args.output_img)
        
    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dict_1d.items()]), index = wave_id)
    df.index.name = 'wave_id'
    df.to_csv(args.output)


