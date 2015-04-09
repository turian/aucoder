#!/usr/bin/python

import argparse
import os.path
import cPickle
import tempfile

from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
from scikits.samplerate import resample
import numpy as n
from pydub import AudioSegment

# We can't work with files that don't have this SAMPLERATE
# TODO convert everything to same samplerate
SAMPLERATE = 44100.

def filename_to_mfcc_frames(filename, winlen, winstep):
    samplerate = SAMPLERATE
    opts = {"samplerate": samplerate,
            "winlen": winlen,
            "winstep": winstep,
            "numcep": 13,
            "nfilt": 26,
            "nfft": 512
            }
    cache_filename = filename + "." + "_".join("%s=%s" % (k, v) for k, v in sorted(opts.items())) + ".pkl"

    if not os.path.exists(cache_filename):
        print "No cached version for %s" % filename
        mfcc_feat = perform_mfcc_on_filename(filename, opts)
        if mfcc_feat is not None:
            cPickle.dump(mfcc_feat, open(cache_filename, "wb"))
            print "Wrote cache to %s" % cache_filename
    else:
        print "Reading cache from %s" % cache_filename
        mfcc_feat = cPickle.load(open(cache_filename, "rb"))
    if mfcc_feat is not None:
        print "%s has MFCC with shape %s" % (filename, repr(mfcc_feat.shape))
    return mfcc_feat

def perform_mfcc_on_filename(filename, opts):
    (samplerate, sig) = read_audio_to_numpy(filename)
    opts['samplerate'] = samplerate
    if sig.ndim > 1:
        # Mix to mono
        # TODO: Multi-channel
        nchannels = sig.shape[1]
        sig = n.mean(sig, axis=1)
    else:
        nchannels = 1
    print "Read %s with sample rate %s, #channels = %d" % (filename, samplerate, nchannels)
    
    if (samplerate != SAMPLERATE):
        sig = resample(sig, SAMPLERATE/samplerate, 'linear')
        print("Resampled")

    mfcc_feat = mfcc(sig, **opts)
    return mfcc_feat

def read_audio_to_numpy(filename):
    if filename.endswith(".mp3"):
        song = AudioSegment.from_mp3(filename)
        filename = filename.replace(".mp3", ".wav")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav")
        song.export(tmp.name, format="wav")
        print "Temporary export to %s" % tmp.name

        (samplerate,signal) = wav.read(tmp.name)
        tmp.close()
    else:
        assert filename.endswith(".wav")
        (samplerate,signal) = wav.read(filename)
    return (samplerate,signal)

# For the input file, find frames that are nearest in the corpus.
# Return a list of the following format:
#   (input frame start sec, input frame end sec, corpus filename, corpus frame start sec, corpus frame end sec)
def find_nearest_frames(input_filename, corpus_filenames, winlen, winstep):
    input_mfcc = filename_to_mfcc_frames(input_filename, winlen, winstep)
    input_nframes = input_mfcc.shape[0]

    corpus = []
    for corpus_filename in corpus_filenames:
        corpus_mfcc = filename_to_mfcc_frames(corpus_filename, winlen, winstep)
        if corpus_mfcc is not None:
            corpus.append((corpus_filename, corpus_mfcc))

    # For each frame, find the nearest frame
    near_frames = []
    for frame_idx in range(min(1000, input_nframes)): #range(nframes):
        this_frame = input_mfcc[frame_idx]
        best_frames = []
        for (corpus_filename, corpus_mfcc) in corpus:
            # Don't allow it to use the same exact frame
            if input_filename == corpus_filename: ignore_frame_idx = frame_idx
            else: ignore_frame_idx = None
            near_idx, near_dist = find_nearest_frame_for_one(this_frame, corpus_mfcc, ignore_frame_idx)
            best_frames.append((near_dist,
                                winstep * frame_idx,
                                winstep * frame_idx + winlen,
                                corpus_filename,
                                winstep * near_idx,
                                winstep * near_idx + winlen))
        best_frames.sort()
        print best_frames[0]
        near_frames.append(best_frames[0][1:])
    return near_frames
      
def find_nearest_frame_for_one(this_frame, corpus_mfcc, ignore_frame_idx):
    # Sum of squared distances (euclidean) against every frame:
    frame_dist = n.square(corpus_mfcc - this_frame).sum(axis=1)
    dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist()) if idx != ignore_frame_idx]
    dist_idx.sort()
    
    near_frame_dist = dist_idx[0][0]
    near_frame_idx = dist_idx[0][1]
    return near_frame_idx, near_frame_dist

# Simple version of redub, that assumes all frame_locations are contiguous
# Frame locations has the following format
#   (input frame start sec, input frame end sec, corpus filename, corpus frame start sec, corpus frame end sec)
def redub(frame_locations, output_filename):
    fragments = []
    for (write_start_sec, write_end_sec, corpus_filename, corpus_start_sec, corpus_end_sec) in frame_locations:
        fragments.append(get_audiosegment(corpus_filename, corpus_start_sec, corpus_end_sec))
    newsong = fragments[0]
    for f in fragments[1:]: newsong += f
    print "Composed %d fragments" % len(fragments)
    newsong.export(output_filename, format="mp3")
    print "Wrote new song to %s" % output_filename

def get_audiosegment(filename, start_sec, end_sec):
    start_ms = int(start_sec * 1000 + 0.5)
    end_ms = int(end_sec * 1000 + 0.5)
    return full_audiosegment(filename)[start_ms:end_ms]

full_audiosegment_cache = {}
def full_audiosegment(filename):
    global full_audiosegment_cache
    if filename not in full_audiosegment_cache:
        full_audiosegment_cache[filename] = AudioSegment.from_mp3(filename)
        print "Read audio from %s" % filename
    return full_audiosegment_cache[filename]

# Version of redub that is slow, but allows files to overlap
def redub_overlay(input_filename, frame_locations, output_filename):
    song = AudioSegment.from_mp3(input_filename)
    print "Read audio from %s" % input_filename

    newsong = AudioSegment.silent(duration=len(origsong))
    for (pos, start_sec, end_sec) in frame_locations:
        pos_ms = int(pos * 1000 + 0.5)
        start_ms = int(start_sec * 1000 + 0.5)
        end_ms = int(end_sec * 1000 + 0.5)
        print (pos_ms, start_ms, end_ms)

        fragment = song[start_ms:end_ms]
        newsong= newsong.overlay(fragment, position=pos_ms)

    newsong.export(output_filename, format="mp3")
    print "Wrote new song to %s" % output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aucode a sound.')
    parser.add_argument('-i', '--input', help='Input audio signal to be covered (mp3)')
    parser.add_argument('-o', '--output', help='Output filename (mp3)')
    parser.add_argument('--winlen', default=250, help='Frame length, in ms')
    parser.add_argument('--winstep', help='Frame step, in ms (= frame length by default)')
    parser.add_argument('-c', '--corpus', help='Audio file(s) to use as samples (mp3)', nargs='*')

    args = parser.parse_args()
    winlen = float(args.winlen) / 1000.0
    winstep = float(args.winstep or args.winlen) / 1000.0

    assert args.input.endswith(".mp3")
    for c in args.corpus:
        assert c.endswith(".mp3")
    assert args.output.endswith(".mp3")

    frame_locations = find_nearest_frames(args.input, args.corpus, winlen, winstep)
    redub(frame_locations, args.output)
