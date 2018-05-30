# wavescalephant Project

## Information on DSPP data files

The simulation reproduce 10 seconds of Slow Wave activity for a grid of cortical columns of size 24 by 24.

Each column is composed of 1250 neurons, divided in three sub-populations:

-          250 excitatory of foreground (the ones that really express slow waves)

-          750 excitatory of background (that are used to sustain the system activity)

-          250 inhibitory

Neurons are connected using an exponential low, which allows to reach target neurons in a mean radius of about 4 cortical columns, for a total of about 1200 synapses per neuron.



There are two DPSNN output files:

-          a spike file (neuron ID plus time of firing)

-          a rate file binned at 5 milliseconds

A bit of explanation for the rate file:

each row contains the binned time in millisec (5, 10, 15,…) plus the rate of each subpopulation in each column of the grid. This means that for a 24by24 grid there are 24x24x3 rate values, plus the first value indicating the time.

Neuron IDs: Consecutive numbering, 1-1250 are neurons from column 1, 1-250 are from foreground, 251-1000 are background, and 1001-1250 are inhibitory.


## Information on NEST files

For comparison, I also run the same simulation in NEST and you can find, in the corresponding NEST folder, the *.gdf files.

Each .gdf file contains neuron IDs of the form
X, 2X, 3X,...
where X is likely the process number. 

For the NEST simulation I’ve used 144 processes, so that I collected 144 output files.

Neuron IDs: ?
