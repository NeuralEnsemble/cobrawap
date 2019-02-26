import argparse
import numpy as np
import scipy as sc



if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs=1, type=str)
    CLI.add_argument("--data",      nargs=1, type=str)
    CLI.add_argument("--freq_band",  nargs=2, type=float)
    CLI.add_argument("--detrending_order", nargs=1, type=int, default=2)
    CLI.add_argument("--psd_num_seg",  nargs=1, type=int)
    CLI.add_argument("--psd_overlap",  nargs=1, type=float)

    args = CLI.parse_args()

    segment = load_segment(args.data[0])

    logMUA_segment = build_logMUA_segment(segment,
                                          freq_band=args.freq_band,
                                          detrending_order=args.detrending_order[0],
                                          psd_num_seg=args.psd_num_seg[0],
                                          psd_overlap=args.psd_overlap[0])
    save(logMUA_segment, args.output[0])