#!/usr/bin/python

import argparse

from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy as n

parser = argparse.ArgumentParser(description='Aucode a sound.')
parser.add_argument('--input', help='Input audio signal to be covered')
#parser.add_argument('--corpus', help='MP3 of audio to use as samples')

args = parser.parse_args()

(rate,sig) = wav.read(args.input)

# TODO convert everything to same samplerate

print "Read %s with sample rate %s" % (args.input, rate)

mfcc_feat = mfcc(sig,rate)

print "Created MFCC with shape", mfcc_feat.shape

nframes = mfcc_feat.shape[0]

for frame_idx in range(nframes):
    this_frame = mfcc_feat[frame_idx]
    
    # Sum of squared distances (euclidean) against every frame:
    frame_dist = n.sqrt(n.square(mfcc_feat - this_frame).sum(axis=1))
    # Remove the frame corresponding to this index
    dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist()) if idx != frame_idx]
    dist_idx.sort()

    near_frame_dist = dist_idx[0][0]
    near_frame_idx = dist_idx[0][1]

    print "Nearest frame to frame #%d is frame #%d (dist = %.3f)" % (frame_idx, near_frame_idx, near_frame_dist)
