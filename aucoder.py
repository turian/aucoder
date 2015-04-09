#!/usr/bin/python

import argparse

parser = argparse.ArgumentParser(description='Aucode a sound.')
parser.add_argument('--input', help='Input audio signal to be covered')
parser.add_argument('--corpus', help='MP3 of audio to use as samples')

args = parser.parse_args()
print args.input
