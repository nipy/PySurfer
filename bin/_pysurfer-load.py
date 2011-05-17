#! /usr/bin/env python
"""
This is the intermediate (hidden) command-line interface script
for PySurfer. It is executed as a run-time script with IPython
by the pysurfer executable, and it is in this script that the
visualiztion environment is actually set up.

"""
import sys
from surfer import Brain
from surfer._commandline import parser

args = parser.parse_args(sys.argv[1].split())

# Load  up the figure and underlying brain object
b = Brain(args.subject_id, args.hemi, args.surf, args.curv, args.title)

# Maybe load some morphometry
if args.morphometry is not None:
    b.add_morphometry(args.morphometry)

# Maybe load an overlay
if args.overlay is not None:
    if args.range is not None:
        args.min, args.max = args.range

    b.add_overlay(args.overlay, args.min, args.max, args.sign)

# Also point brain at the Brain() object
brain = b

# It's nice to have mlab in the namespace, but we'll import it
# after the other stuff so getting usage is not interminable
from enthought.mayavi import mlab

# Now clean up the namespace a bit
del parser, args
