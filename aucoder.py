#!/usr/bin/python

import argparse
import os.path
import cPickle
import tempfile
import random
import hashlib

from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
from scikits.samplerate import resample
import numpy as n
from pydub import AudioSegment
from itertools import tee, izip
from annoy import AnnoyIndex

# We can't work with files that don't have this desired_samplerate
desired_samplerate = 44100
desired_channels = 2            # We expect this for corpus files
FORCE_RESAMPLE = False          # This can be really slow

#ANN_NTREES = 100
ANN_NTREES = 10
ANN_CANDIDATES = 10

IGNORE_SAME_FRAME = True

SECONDS_OUTPUT = 30.0
#SECONDS_OUTPUT = None

SEED = 0
random.seed(SEED)

def filename_to_mfcc_frames(filename, winlen, winstep):
    samplerate = desired_samplerate
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
        cPickle.dump(mfcc_feat, open(cache_filename, "wb"))
        print "\tWrote cache to %s" % cache_filename
    else:
#        print "\tReading cache from %s" % cache_filename
        mfcc_feat = cPickle.load(open(cache_filename, "rb"))
        if mfcc_feat is None:
            print "No MFCC for %s, perhaps has wrong samplerate" % filename
#    if mfcc_feat is not None:
#        print "%s has MFCC with shape %s" % (filename, repr(mfcc_feat.shape))
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
    
    if (samplerate != desired_samplerate and not FORCE_RESAMPLE):
        print "%s has the wrong samplerate, ignoring" % filename
        return None

    if (samplerate != desired_samplerate and FORCE_RESAMPLE):
        origsig = sig
        sig = resample(origsig, 1.0 * desired_samplerate/samplerate, 'sinc_best')
        print "Resampled file from rate %d to rate %d, shape %s to %s" % (samplerate, desired_samplerate, origsig.shape, sig.shape)

    mfcc_feat = mfcc(sig, **opts)
    print "Computed mfccs on %s" % filename
    return mfcc_feat


full_audiosegment_wave_cache = {}
def read_audio_to_numpy(filename):
    global full_audiosegment_wave_cache
    if filename not in full_audiosegment_wave_cache:
        (samplerate,signal) = read_audio_to_numpy_uncached(filename)
        assert samplerate == desired_samplerate
        full_audiosegment_wave_cache[filename] = (samplerate, signal)
        print "\tRead audio from %s" % filename
    return full_audiosegment_wave_cache[filename]

def read_audio_to_numpy_uncached(filename):
    if filename.endswith(".mp3"):
        song = AudioSegment.from_mp3(filename)
        filename = filename.replace(".mp3", ".wav")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav")
        song.export(tmp.name, format="wav")
#        print "Temporary export to %s" % tmp.name

        (samplerate,signal) = wav.read(tmp.name)
        tmp.close()
    else:
        assert filename.endswith(".wav")
        (samplerate,signal) = wav.read(filename)
    return (samplerate,signal)

# For the input file, find frames that are nearest in the corpus.
# Return a list of the following format:
#   (input frame start sec, input frame end sec, corpus filename, corpus frame start sec, corpus frame end sec)
annoy_mfcc_index, annoy_mfcc_list = None, None
def find_nearest_frames(input_filename, corpus_filenames, winlen, winstep):
    global annoy_mfcc_index, annoy_mfcc_list
    input_mfcc = filename_to_mfcc_frames(input_filename, winlen, winstep)
    input_nframes = input_mfcc.shape[0]
    dimension = input_mfcc.shape[1]

    corpus = []
    total_length = 0
    for corpus_filename in corpus_filenames:
        corpus_mfcc = filename_to_mfcc_frames(corpus_filename, winlen, winstep)
        total_length += corpus_mfcc.shape[0]
        if corpus_mfcc is not None:
            corpus.append((corpus_filename, corpus_mfcc))
    print "Corpus has %.2f hr, %d frames" % (total_length * winstep / 60 / 60, total_length)

    #Build an AnnoyIndex
    annoy_mfcc_index, annoy_mfcc_list = build_annoy_index(corpus, dimension, winlen, winstep)

    # For each frame, find the nearest frame
    dists = []
    near_frames = []
    approx_dist_error = []
    if SECONDS_OUTPUT:
        nframes_to_process = min(int(SECONDS_OUTPUT / winstep), input_nframes)
    else:
        nframes_to_process = input_nframes
    for frame_idx in range(nframes_to_process):
        this_frame = input_mfcc[frame_idx]
        (near_dist, corpus_filename, near_idx) = \
            find_nearest_frame_annoy(this_frame, input_filename, frame_idx, corpus)
        # Compute error against exhaustive
        if len(approx_dist_error) < 30 and random.random() < 0.01:
            (near_dist2, corpus_filename2, near_idx2) = \
                find_nearest_frame_exhaustive(this_frame, input_filename, frame_idx, corpus)
            assert near_dist2 <= near_dist
            approx_dist_error.append(near_dist - near_dist2)

        best_frame = (near_dist,
                      winstep * frame_idx,
                      winstep * frame_idx + winlen,
                      corpus_filename,
                      winstep * near_idx,
                      winstep * near_idx + winlen)
        dists.append(best_frame[0])
        near_frames.append(best_frame[1:])
    dists = n.array(dists)
    print "DISTANCE median=%.3f, mean=%.3f" % (n.median(dists), n.mean(dists))
    print "Average error (distance) because of approximate nearest neighbors: %.3f" % n.mean(approx_dist_error)
    return near_frames

# Exhaustive technique to find the nearest frame
# Return (distance, corpus file, corpus frame idx)
def find_nearest_frame_exhaustive(this_frame, input_filename, input_frame_idx, corpus):
    best_frames = []
    for (corpus_filename, corpus_mfcc) in corpus:
        # Don't allow it to use the same exact frame
        if input_filename == corpus_filename and IGNORE_SAME_FRAME: ignore_frame_idx = input_frame_idx
        else: ignore_frame_idx = None
        near_idx, near_dist = find_nearest_frame_for_one_with_one_corpus_file(this_frame, corpus_mfcc, ignore_frame_idx)
        best_frames.append((near_dist, corpus_filename, near_idx))
    best_frames.sort()
    return best_frames[0]

def find_nearest_frame_for_one_with_one_corpus_file(this_frame, corpus_mfcc, ignore_frame_idx):
    # Sum of squared distances (euclidean) against every frame:
    frame_dist = n.sqrt(n.square(corpus_mfcc - this_frame).sum(axis=1))
    dist_idx = [(dist, idx) for (idx, dist) in enumerate(frame_dist.tolist()) if idx != ignore_frame_idx]
    dist_idx.sort()
    
    near_frame_dist = dist_idx[0][0]
    near_frame_idx = dist_idx[0][1]
    return near_frame_idx, near_frame_dist

# Approx nearest neighbor technique to find the nearest frame
# Return (distance, corpus file, corpus frame idx)
def find_nearest_frame_annoy(this_frame, input_filename, input_frame_idx, corpus):
    nearest_neighbors = annoy_mfcc_index.get_nns_by_vector(this_frame.tolist(), ANN_CANDIDATES)
    candidates = []
    for nearest_neighbor in nearest_neighbors:
        corpus_filename, near_idx = annoy_mfcc_list[nearest_neighbor]
        if IGNORE_SAME_FRAME and corpus_filename == input_filename and near_idx == input_frame_idx: continue
        corpus_mfcc = None
        for (filename, mfcc) in corpus:
            if filename == corpus_filename: corpus_mfcc = mfcc
        near_dist = n.sqrt(n.square(corpus_mfcc[near_idx] - this_frame).sum())
        candidates.append((near_dist, corpus_filename, near_idx))
    candidates.sort()
    # If nothing is returned, try ANN_CANDIDATES > 1
    # Otherwise, we post filter and disallow you to return the frame being used to search
    return candidates[0]

def build_annoy_index(corpus, dimension, winlen, winstep):
    print "Adding to Annoy index"
    index = AnnoyIndex(dimension, "euclidean")
    mfcc_list = []
    i = 0
    for filename, frames in corpus:
#        print filename, frames.shape
        for index_in_file, mfcc in enumerate(frames):
            mfcc_list.append((filename, index_in_file))
            index.add_item(i, mfcc.tolist())
            assert mfcc_list[i] == (filename, index_in_file)
            i += 1

    opts = {"samplerate": desired_samplerate,
            "winlen": winlen,
            "winstep": winstep,
            "numcep": 13,
            "nfilt": 26,
            "nfft": 512,
            "ntrees": ANN_NTREES
            }
    cache_filename = "annoy_index_" + hashlib.md5(str([filename for filename, frames in corpus])).hexdigest() + "." + "_".join("%s=%s" % (k, v) for k, v in sorted(opts.items())) + ".tree"
    
    if not os.path.exists(cache_filename):
        print "Building Annoy index with %d trees" % ANN_NTREES
    #    index.build(-1)
        index.build(ANN_NTREES)
        index.save(cache_filename)
        print "\tWrote cache to %s" % cache_filename
    else:
        print "\tReading cache from %s" % cache_filename
        index.load(cache_filename)
    return index, mfcc_list

def get_audiosegment_wave(filename, start, end):
    return read_audio_to_numpy(filename)[1][start:end]

def sec2sample(sec):
    return int(sec * desired_samplerate + 0.5)

def redub_overlay_wave(frame_locations, output_filename):
    print ""
    print "Beginning redub_overlay_wave to %s with %d frames" % (output_filename, len(frame_locations))
    start_points = set(sec2sample(frame[0]) for frame in frame_locations)
    end_points = set(sec2sample(frame[1]) for frame in frame_locations)
    cut_points = sorted(start_points.union(end_points))
    cuts = [x for x in window(cut_points, 2)]

    updated_frame_locations = []
    for (write_start_sec, write_end_sec, corpus_filename, corpus_start_sec, corpus_end_sec) in frame_locations:
        write_start = sec2sample(write_start_sec)
        write_end = sec2sample(write_end_sec)
        corpus_start = sec2sample(corpus_start_sec)
        corpus_end = sec2sample(corpus_end_sec)
        updated_frame_locations.append((write_start, write_end, corpus_filename, corpus_start, corpus_end))
    updated_frame_locations.sort()
    assert(len(updated_frame_locations) == len(frame_locations))
    frame_locations = updated_frame_locations
        

    fragments = []
    for (j, (cut_start, cut_end)) in enumerate(cuts):
        if j % 100 == 0:
            print "Making cut %d of %d" % (j, len(cuts))
        cut_length = cut_end - cut_start
        this_fragments = []
        # TODO: this nested loop can be a bit slow, but we're always searching in
        #       one direction. We could speed this up with some trickery.
        for (write_start, write_end, corpus_filename, corpus_start, corpus_end) in frame_locations:
            if write_start >= cut_end or write_end <= cut_start:
                continue

            desired_cut_start = max(write_start, cut_start)
            desired_cut_end   = min(write_end, cut_end)
            assert desired_cut_end >= desired_cut_start

            if (desired_cut_end - desired_cut_start) != cut_length:
                print "Weird. Skipping this cut, can't get the right size"
                continue

            actual_start = corpus_start + (desired_cut_start - write_start)
            actual_end   = actual_start + desired_cut_end - desired_cut_start
            assert actual_end >= actual_start

            segment = get_audiosegment_wave(corpus_filename, actual_start, actual_end)
            this_fragments.append(segment)

        to_avg = [n.zeros((cut_length, desired_channels))]
        for f in this_fragments:
            assert f.shape[1] == desired_channels
            if f.shape[0] != cut_length:
                print "Weird. Extracted cut of the wrong size. Expected = %d, received = %d" % (cut_length, f.shape[0])
                continue
            else:
                to_avg.append(f)
        total = to_avg[0].astype("float")
        for t in to_avg[1:]:
            assert(total.shape == t.shape)
            total += t.astype("float")
        total /= len(to_avg)
        assert(total.shape[0] == cut_length)
        fragments.append(total.astype("int16"))
    
    newsong = n.vstack(fragments)
    print "Composed %d fragments into %s" % (len(fragments), newsong.shape)
    wav.write(filename = output_filename, rate = desired_samplerate, data = newsong)
    print "Wrote new song to %s" % output_filename

def window(iterable, size):
    iters = tee(iterable, size)
    for i in xrange(1, size):
        for each in iters[i:]:
            next(each, None)
    return izip(*iters)

def submp3(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".mp3"):
                 yield (os.path.join(root, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aucode a sound.')
    parser.add_argument('-i', '--input', help='Input audio signal to be covered (mp3)')
    parser.add_argument('--winlen', default=250, help='Frame length, in ms')
    parser.add_argument('--winstep', help='Frame step, in ms (= frame length by default)')
    parser.add_argument('-c', '--corpus', help='Audio file(s) to use as samples (mp3), or the name of a directory', nargs='*')

    args = parser.parse_args()
    winlen = float(args.winlen) / 1000.0
    winstep = float(args.winstep or args.winlen) / 1000.0

    assert args.input.endswith(".mp3")

    corpus = []
    for c in args.corpus:
        if os.path.isdir(c):
            for f in submp3(c):
                corpus.append(f)
        else:
            assert c.endswith(".mp3")
            corpus.append(c)
#    print "Corpus =", corpus

    def basefile(f):
        if os.path.isdir(f):
            return os.path.split(os.path.normpath(f))[1]
        else:
            return os.path.splitext(os.path.split(os.path.normpath(f))[1])[0]

    l = ""
    if len(args.corpus) > 1: l == str(len(args.corpus))
    output = os.path.join("output", basefile(args.input) + " - " + basefile(args.corpus[0]) + l + " winlen=%f winstep=%f.wav" % (winlen, winstep))
    print "Using output =", output
    assert output.endswith(".wav")

    frame_locations = find_nearest_frames(args.input, corpus, winlen, winstep)
    redub_overlay_wave(frame_locations, output)
