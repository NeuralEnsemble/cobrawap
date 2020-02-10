import os
import sys
import argparse
import quantities as pq
import neo
import numpy as np
import scipy

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--data",          nargs='?', type=str)
    CLI.add_argument("--output",  nargs='?', type=str)
    CLI.add_argument("--t_start",    nargs='?', type=float)
    CLI.add_argument("--t_stop",  nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        blk = io.read_block()

    asig = blk.segments[0].analogsignals[0]

    t_start = max([args.t_start, asig.t_start.rescale('s').magnitude])
    t_stop  = min([args.t_stop, asig.t_stop.rescale('s').magnitude])

    for i, asig in enumerate(blk.segments[0].analogsignals):
        blk.segments[0].analogsignals[i] = asig.time_slice(t_start=t_start*pq.s,
                                                           t_stop=t_stop*pq.s)

    for i, evt in enumerate(blk.segments[0].events):
        blk.segments[0].events[i] = evt.time_slice(t_start=t_start*pq.s,
                                                   t_stop=t_stop*pq.s)


    with neo.NixIO(args.output) as io:
        io.write(blk)
