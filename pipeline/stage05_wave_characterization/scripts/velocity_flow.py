import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import scipy
import pandas as pd
from utils.io import load_neo, save_plot
from utils.parse import none_or_str


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
    asig = [a for a in block.segments[0].analogsignals if a.name == 'Optical Flow'][0]

    wave_ids = np.unique(evts.labels)
    velocities = np.empty((len(wave_ids),2))
    velocities.fill(np.nan)
    v_unit = (asig.annotations['spatial_scale'].units/asig.sampling_period.units).dimensionality.string

    for i, wave_id in enumerate(wave_ids):
        if not wave_id == '-1':
            idx = np.where(evts.labels == str(wave_id))[0]
            trigger_channels = evts.array_annotations['channels'][idx]
            trigger_times = evts.times[idx]
            t_idx = np.array([np.argmax(asig.times >= t) for t in trigger_times])
            wave_vectors = asig.as_array()[t_idx, trigger_channels]

            dxy = np.abs(wave_vectors)
            dxy = dxy * asig.annotations['spatial_scale']/asig.sampling_period
            velocities[i,0] = np.mean(dxy)
            velocities[i,1] = np.std(dxy)

            save_path = os.path.splitext(args.output)[0] + f'_wave{wave_id}'
            np.save(save_path+'.npy', dxy)

            fig, ax = plt.subplots()
            sns.histplot(dxy, kde=True, ax=ax)
            ax.set_title(f'wave {wave_id}')
            ax.set_xlabel(f'instantanous velocities [{v_unit}]')
            ax.set_ylabel('')
            save_plot(save_path+'.png')

    # transform to DataFrame
    df = pd.DataFrame(velocities,
                      columns=['velocity_flow', 'velocity_flow_std'],
                      index=wave_ids)
    df['velocity_unit'] = [v_unit]*len(wave_ids)
    df.index.name = 'wave_id'

    df.to_csv(args.output)
