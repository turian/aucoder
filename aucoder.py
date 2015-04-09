#!/usr/bin/python

import argparse
import os.path
import cPickle

from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import numpy as n
from pydub import AudioSegment

# We can't work with files that don't have this SAMPLERATE
# TODO convert everything to same samplerate
SAMPLERATE = 44100

def convert_to_wav(filename):
    if filename.endswith(".mp3"):
        song = AudioSegment.from_mp3(filename)
        filename = filename.replace(".mp3", ".wav")
        song.export(filename, format="wav")
    return filename

def filename_to_mfcc_frames(filename, winlen, winstep):
    samplerate = SAMPLERATE
    opts = {"samplerate": samplerate,
            "winlen": winlen,
            "winstep": winstep,
            "numcep": 13,
            "nfilt": 26,
            "nfft": 512
            }
    cache_filename = filename + "_".join("%s=%s" % (k, v) for k, v in sorted(opts.items())) + ".pkl"

    if not os.path.exists(cache_filename):
        print "No cached version for %s" % filename
        mfcc_feat = perform_mfcc_on_filename(filename, opts)
        cPickle.dump(mfcc_feat, open(cache_filename, "wb"))
        print "Wrote cache to %s" % cache_filename
    else:
        print "Reading cache from %s" % cache_filename
        mfcc_feat = cPickle.load(open(cache_filename, "rb"))
    print "%s has MFCC with shape %s" % (filename, repr(mfcc_feat.shape))
    return mfcc_feat

def perform_mfcc_on_filename(filename, opts):
    (samplerate,sig) = wav.read(filename)
    nchannels = sig.shape[1]
    assert samplerate == SAMPLERATE

    print "Read %s with sample rate %s, #channels = %d" % (filename, samplerate, sig.shape[1])

    # Mix to mono
    # TODO: Multi-channel
    sig = n.mean(sig, axis=1)

    mfcc_feat = mfcc(sig, **opts)
    return mfcc_feat

def find_nearest_frames(input_filename, corpus_filename, winlen, winstep):
    input_mfcc = filename_to_mfcc_frames(input_filename, winlen, winstep)
    input_nframes = input_mfcc.shape[0]
    corpus_mfcc = filename_to_mfcc_frames(corpus_filename, winlen, winstep)
    # For each frame, find the nearest frame
    near_frame_idxs = []
    for frame_idx in range(min(1000, input_nframes)): #range(nframes):
        this_frame = input_mfcc[frame_idx]
        
        # Sum of squared distances (euclidean) against every frame:
        frame_dist = n.square(corpus_mfcc - this_frame).sum(axis=1)
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
    for input_idx, corpus_idx in enumerate(near_frame_idxs):
        frame_locations.append((winstep * input_idx, winstep * corpus_idx, winstep * corpus_idx + winlen))
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

# Version of redub that is slow, but allows files to overlap
def redub_overlay(orig_filename, input_filename, frame_locations, output_filename):
    origsong = AudioSegment.from_wav(orig_filename)
    print "Read audio from %s" % orig_filename

    song = AudioSegment.from_wav(input_filename)
    print "Read audio from %s" % input_filename

    newsong = AudioSegment.silent(duration=len(origsong))
    for (pos, start_sec, end_sec) in frame_locations:
        pos_ms = int(pos * 1000 + 0.5)
        start_ms = int(start_sec * 1000 + 0.5)
        end_ms = int(end_sec * 1000 + 0.5)
        print (pos_ms, start_ms, end_ms)

        fragment = song[start_ms:end_ms]
        newsong= newsong.overlay(fragment, position=pos_ms)

    # Now, overlay the original audio at lower volume
    #origsong = AudioSegment.from_wav(orig_filename)
    #print "Read audio from %s" % orig_filename
    ##origsong = origsong.apply_gain(-10)
    #newsong = newsong.overlay(origsong)

    newsong.export(output_filename, format="mp3")
    print "Wrote new song to %s" % output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aucode a sound.')
    parser.add_argument('--input', help='Input audio signal to be covered (wav or mp3)')
    parser.add_argument('--output', help='Output filename (mp3)')
    parser.add_argument('--winlen', default=250, help='Frame length, in ms')
    parser.add_argument('--winstep', help='Frame step, in ms (= frame length by default)')
    parser.add_argument('--corpus', help='Audio file(s) to use as samples (wav or mp3)', nargs='*')

    args = parser.parse_args()
    input_wav = convert_to_wav(args.input)
    corpus_wav = convert_to_wav(args.corpus[0])
    winlen = float(args.winlen) / 1000.0
    winstep = float(args.winstep or args.winlen) / 1000.0

    frame_locations = find_nearest_frames(input_wav, corpus_wav, winlen, winstep)
    redub(input_wav, corpus_wav, frame_locations, args.output)
