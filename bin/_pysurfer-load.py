#! /usr/bin/env python
import sys
import argparse
sys.path.append("/mindhive/gablab/u/mwaskom/PySurfer")
import surfer as sf

parser = argparse.ArgumentParser()
parser.add_argument("subject_id")
parser.add_argument("hemi", choices=["lh", "rh"])
parser.add_argument("surf")
parser.add_argument("-overlay")
parser.add_argument("-fminmax", nargs=2, default=[2,5])

args = parser.parse_args(sys.argv[1].split())

brain = sf.FSBrain(args.subject_id, args.hemi, args.surf)
if args.overlay is not None:
    brain.add_overlay(args.overlay, *args.fminmax)
