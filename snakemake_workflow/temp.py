import sys
import os
import neo
import numpy as np
sys.path.append('/home/rgutzen/Projects/viziphant/')
from viziphant.rasterplot import rasterplot
import quantities as pq
import matplotlib.pyplot as plt

img_dir = '/home/rgutzen/ProjectsData/wavescalephant/LENS/170110_mouse2_deep/t1'
with neo.NixIO(os.path.join(img_dir, 'up_transitions.nix')) as io:
    ca_ups = io.read_block().segments[0].spiketrains

fig, ax = plt.subplots(figsize=(10,10))
rasterplot(ca_ups, ax=ax)
plt.show()

fp_dir = '/home/rgutzen/ProjectsData/wavescalephant/IDIBAPS/161101_rec07_Spontaneous_RH'
with neo.NixIO(os.path.join(fp_dir, 'UD_states.nix')) as io:
    fp_ups = io.read_block().segments[0].spiketrains

fig, ax = plt.subplots(figsize=(10,10))
rasterplot([up.time_slice(0*pq.s, 10*pq.s) for up in fp_ups], 
           ax=ax, key_list=['channel_ids'], labelkey='channel_ids')
plt.show()
