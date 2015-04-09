#!/usr/bin/python

import argparse

from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy as n
from pydub import AudioSegment

def convert_to_wav(filename):
    if filename.endswith(".mp3"):
        song = AudioSegment.from_mp3(filename)
        filename = filename.replace(".mp3", ".wav")
        song.export(filename, format="wav")
    return filename

def filename_to_mfcc_frames(filename, winlen, winstep):
    # TODO convert everything to same samplerate
    (rate,sig) = wav.read(filename)
    nchannels = sig.shape[1]
    print "Read %s with sample rate %s, #channels = %d" % (filename, rate, sig.shape[1])

    # Mix to mono
    # TODO: Multi-channel
    sig = n.mean(sig, axis=1)

#    # 30 seconds
#    sig = sig[:1323000]
    
    mfcc_feat = mfcc(sig, rate, winlen=winlen, winstep=winstep)
    print "Created MFCC with shape", mfcc_feat.shape
    #nframes = mfcc_feat.shape[0]
    return mfcc_feat

def find_nearest_frames(input_filename, corpus_filename, winlen, winstep):
    input_mfcc = filename_to_mfcc_frames(input_filename, winlen, winstep)
    corpus_mfcc = filename_to_mfcc_frames(corpus_filename, winlen, winstep)
    # For each frame, find the nearest frame
    near_frame_idxs = []
    for frame_idx in range(1000): #range(nframes):
        this_frame = input_mfcc[frame_idx]
        
        # Sum of squared distances (euclidean) against every frame:
        frame_dist = n.sqrt(n.square(corpus_mfcc - this_frame).sum(axis=1))
        # Remove the frame corresponding to this index
        dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist()) if idx != frame_idx]
#        dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist())]
        dist_idx.sort()
    
        near_frame_dist = dist_idx[0][0]
        near_frame_idx = dist_idx[0][1]
    
        print "Nearest frame to frame #%d is frame #%d (dist = %.3f)" % (frame_idx, near_frame_idx, near_frame_dist)
        near_frame_idxs.append(near_frame_idx)

    print near_frame_idxs
    frame_locations = []
    for idx in near_frame_idxs:
        frame_locations.append((winstep * idx, winstep * idx + winlen))
    return frame_locations

def redub(input_filename, frame_locations, output_filename):
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
    newsong.export(output_filename, format="mp3")
    print "Wrote new song to %s" % output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aucode a sound.')
    parser.add_argument('--input', help='Input audio signal to be covered (wav or mp3)')
    parser.add_argument('--corpus', help='Audio file to use as samples (wav or mp3)')
    parser.add_argument('--output', help='Output filename (mp3)')
    parser.add_argument('--winlen', help='Frame length, in ms')
    parser.add_argument('--winstep', help='Frame step, in ms (= frame length by default)')

    args = parser.parse_args()
    input_wav = convert_to_wav(args.input)
    corpus_wav = convert_to_wav(args.corpus)
    winlen = float(args.winlen) / 1000.0
    winstep = float(args.winstep or args.winlen) / 1000.0

    frame_locations = find_nearest_frames(input_wav, corpus_wav, winlen, winstep)
    redub(input_wav, frame_locations, args.output)
