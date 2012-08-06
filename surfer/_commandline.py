"""
This module defines the command-line interface for PySurfer.
It is defined here instead of in either the top level or
intermediate start-up scripts, as it is used in both.

There should be no reason to import this module in an
interpreter session.

"""
import argparse

help_text = """
PySurfer is a package for visualization and interaction with cortical
surface representations of neuroimaging data from Freesurfer.

The command-line program pysurfer is designed to largely replicate
Freesufer's tksurfer command-line interface in the format and style
of arguments it accepts, and, like tksurfer, invoking it will initialize
a visualization in an external window and begin an IPython session in the
terminal, through which the visualization can be manipulated.

The visualization interface is exposed through methods on the `brain'
variable that will exist in IPython namespace when the program finishes
loading. Please see the PySurfer documentation for more information
about how to interact with the Brain object.

"""

parser = argparse.ArgumentParser(prog='pysurfer',
                  usage='%(prog)s subject_id hemisphere surface [options]',
                  formatter_class=argparse.RawDescriptionHelpFormatter,
                  description=help_text)
parser.add_argument("subject_id",
                    help="subject id as in subjects dir")
parser.add_argument("hemi", metavar="hemi", choices=["lh", "rh"],
                    help="hemisphere to load")
parser.add_argument("surf",
                    help="surface mesh (e.g. 'pial', 'inflated')")
parser.add_argument("-no-curv", action="store_false", dest="curv",
                    help="do not display the binarized surface curvature")
parser.add_argument("-morphometry", metavar="MEAS",
                    help="load morphometry file (e.g. thickness, curvature)")
parser.add_argument("-annotation", metavar="ANNOT",
                    help="load annotation (by name or filepath)")
parser.add_argument("-label",
                    help="load label (by name or filepath")
parser.add_argument("-borders", action="store_true",
                    help="only show label/annot borders")
parser.add_argument("-overlay", metavar="FILE",
                    help="load scalar overlay file")
parser.add_argument("-range", metavar=('MIN', 'MAX'), nargs=2,
                    help="overlay threshold and saturation point")
parser.add_argument("-min", type=float,
                    help="overlay threshold")
parser.add_argument("-max", type=float,
                    help="overlay saturation point")
parser.add_argument("-sign", default="abs", choices=["abs", "pos", "neg"],
                    help="overlay sign")
parser.add_argument("-name",
                    help="name to use for the overlay")
parser.add_argument("-size",
                    help="size of the display window (in pixels)")
parser.add_argument("-background", metavar="COLOR",
                    help="background color for display")
parser.add_argument("-cortex", metavar="COLOR",
                    help="colormap for binary cortex curvature")
parser.add_argument("-title",
                    help="title to use for the figure")
