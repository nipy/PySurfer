#! /usr/bin/env python
import sys
import argparse
from surfer import Brain

parser = argparse.ArgumentParser()
parser.add_argument("subject_id")
parser.add_argument("hemi", choices=["lh", "rh"])
parser.add_argument("surf")
parser.add_argument("-overlay")
parser.add_argument("-fminmax", nargs=2, default=(2, 5))
parser.add_argument("-sign", default="abs")

args = parser.parse_args(sys.argv[1].split())

brain = Brain(args.subject_id, args.hemi, args.surf)
if args.overlay is not None:
    brain.add_overlay(args.overlay, args.fminmax, args.sign)
