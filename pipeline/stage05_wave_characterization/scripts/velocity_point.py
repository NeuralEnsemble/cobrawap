import neo
import numpy as np
import math
import quantities as pq
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import pandas as pd
from utils import load_neo, none_or_str, save_plot


def velocity(wave_times, DIM_X, DIM_Y):

    #spatial_scale = SpatialScale*pq.mm #wave_times.annotation['spatial_scale']
    spatial_scale = wave_times.annotations['spatial_scale']

    v_unit = (spatial_scale.units/wave_times.times.units).dimensionality.string

    #===========================WAVE VELOCITY===========================
    Labels = [int(float(i)) for i in wave_times.labels]
    wave_labels = np.unique(Labels) # wave indexes
    point_velocity = []
    point_direction = []
    point_label = []
    pos_x = []
    pos_y = []
    
    mean_velocity = []
    
    print('wave lab 0', np.shape(wave_labels))
    #ciclo su tutte le onde di una data raccolta dati
    for w in wave_labels: # for each wave
        print('W',w)
        w_idx = np.where(Labels == w)[0]
        T = np.sort([wave_times.times.magnitude[i] for i in w_idx])
        pixel_idx = wave_times.array_annotations['channels'][w_idx]
        x_idx = wave_times.array_annotations['x_coords'][w_idx]
        y_idx = wave_times.array_annotations['y_coords'][w_idx]
        
        #indici = ord_idx[begin:end]
        grid = np.zeros((DIM_Y,DIM_X))

        #creo la griglia
        for k in range(0,len(T)): # for each point
            x = x_idx[k]
            y =  y_idx[k]
            grid[int(x), int(y)] = T[k]


        count = 0
        #velocity = 0
        velocity = []
        Tx_temp = 0
        Ty_temp = 0

        for x in range(1,DIM_Y-1):
            for y in range(1,DIM_X-1):

                if ( grid[x+1,y] != 0  and grid[x-1,y] != 0 and grid[x,y+1] != 0  and grid[x,y-1] != 0) and ( grid[x+1,y] != np.nan  and grid[x-1,y] != np.nan and grid[x,y+1] != np.nan  and grid[x,y-1] != np.nan):
                    Tx_temp=((grid[x+1,y] - grid[x-1,y])/(2*spatial_scale))
                    Ty_temp=((grid[x,y+1] - grid[x,y-1])/(2*spatial_scale))
                    velocity.append(1/math.sqrt(Tx_temp**2+Ty_temp**2))
                    point_velocity.append(1/math.sqrt(Tx_temp**2+Ty_temp**2))
                    point_direction.append(float(y+1-(y-1))/float(x+1-(x-1)))
                    point_label.append(int(w))
                    pos_x.append(x)
                    pos_y.append(y)
                    count +=1

        if count != 0:
            mean_velocity.append(np.mean(velocity))
            print(mean_velocity)
    plt.figure()
    plt.hist(mean_velocity)
    plt.xlabel('velocity [mm/s]')
    
    
    # transform to DataFrame
    df1 = pd.DataFrame({'Mean velocity': mean_velocity,
                        'velocity_unit': [v_unit]*len(mean_velocity)})
    
    df2 = pd.DataFrame({'Wave Label': point_label,
                        'Point velocity': point_velocity,
                        'Point direction': point_direction,
                        'x coord': pos_x,
                        'y coord': pos_y})
    df = pd.concat([df1, df2], ignore_index=False, axis=1) 
    #df['velocity_unit'] = [v_unit]*len(wave_ids)
    #df.index.name = 'wave_id'

    return(df)

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
    DIM_X = 50
    DIM_Y = 50

    #SpatialScale = 0.005
    
    evts = [ev for ev in block.segments[0].events if ev.name == 'Wavefronts'][0]
    
    print('Spatial', evts.annotations)
    SpatialScale = evts.annotations['spatial_scale']

    #print('ops', evts.annotations['spatial_scale'])
    velocities_df = velocity(evts, DIM_X, DIM_Y)
    print('velocities')
    
    if args.output_img is not None:
        save_plot(args.output_img)

    velocities_df.to_csv(args.output)
