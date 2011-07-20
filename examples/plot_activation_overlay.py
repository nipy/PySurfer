"""
Display fMRI Activation
=======================

Load a statistical overlay on the inflated surface.

"""

print __doc__

import os.path as op
from surfer import Brain
# 
brain = Brain("fsaverage", "lh", "inflated")

overlay_file = op.join(op.dirname(__file__), "data/lh.sig.nii.gz")

brain.add_overlay(overlay_file, min=2, max=10, sign="abs")

brain.overlay["sig"].remove()

brain.add_overlay(overlay_file, min=4, max=15, sign="pos")

brain.show_view("parietal")
