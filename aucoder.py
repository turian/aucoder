#!/usr/bin/python

import argparse

from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy as n
from pydub import AudioSegment

#WINLEN = 0.025      # 25 ms
WINLEN = 0.25      # 250 ms
WINSTEP = WINLEN    # Don't allow them to overlap

def filename_to_mfcc_frames(filename):
    # TODO convert everything to same samplerate
    (rate,sig) = wav.read(filename)
    nchannels = sig.shape[1]
    print "Read %s with sample rate %s, #channels = %d" % (filename, rate, sig.shape[1])

    # Mix to mono
    # TODO: Multi-channel
    sig = n.mean(sig, axis=1)

#    # 30 seconds
#    sig = sig[:1323000]
    
    mfcc_feat = mfcc(sig, rate, winlen=WINLEN, winstep=WINSTEP)

    print "Created MFCC with shape", mfcc_feat.shape
    #nframes = mfcc_feat.shape[0]
    return mfcc_feat

def find_nearest_frames(input_filename, corpus_filename):
    input_mfcc = filename_to_mfcc_frames(input_filename)
    corpus_mfcc = filename_to_mfcc_frames(corpus_filename)
    # For each frame, find the nearest frame
    near_frame_idxs = []
    for frame_idx in range(1000): #range(nframes):
        this_frame = input_mfcc[frame_idx]
        
        # Sum of squared distances (euclidean) against every frame:
        frame_dist = n.sqrt(n.square(corpus_mfcc - this_frame).sum(axis=1))
        if input_filename == corpus_filename:
            # Remove the frame corresponding to this index
            dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist()) if idx != frame_idx]
        else:
            dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist())]
        dist_idx.sort()
    
        near_frame_dist = dist_idx[0][0]
        near_frame_idx = dist_idx[0][1]
    
        print "Nearest frame to frame #%d is frame #%d (dist = %.3f)" % (frame_idx, near_frame_idx, near_frame_dist)
        near_frame_idxs.append(near_frame_idx)

    print near_frame_idxs
    frame_locations = []
    for idx in near_frame_idxs:
        frame_locations.append((WINSTEP * idx, WINSTEP * idx + WINLEN))
    return frame_locations

def redub(orig_filename, input_filename, frame_locations, output_filename):
    song = AudioSegment.from_wav(input_filename)
    print "Read audio from %s" % input_filename
    fragments = []
    for (start_sec, end_sec) in frame_locations:
        start_ms = int(start_sec * 1000 + 0.5)
        end_ms = int(end_sec * 1000 + 0.5)
        fragment = song[start_ms:end_ms]
        fragments.append(fragment)
    newsong = fragments[0]
    for f in fragments[1:]: newsong += f

    # Now, overlay the original audio at lower volume
    origsong = AudioSegment.from_wav(orig_filename)
    print "Read audio from %s" % orig_filename
#    origsong = origsong.apply_gain(-10)
    newsong = newsong.overlay(origsong)

    newsong.export(output_filename, format="mp3")
    print "Wrote new song to %s" % output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aucode a sound.')
    parser.add_argument('--input', dest='input', help='Input audio signal to be covered (wav)')
    parser.add_argument('--corpus', help='Audio file to use as samples (wav)')
    parser.add_argument('--output', dest='output', help='Output filename (mp3)')
#    parser.add_argument('--winlen', dest='winlen', default=0.025, help='Frame window length')
#    parser.add_argument('--winstep', dest='winstep', default=0.025, help='Frame window length')

    args = parser.parse_args()
    frame_locations = find_nearest_frames(args.input, args.corpus)
    redub(args.input, args.corpus, frame_locations, args.output)
