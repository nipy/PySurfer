#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat patch demo.
"""

from surfer import Brain
from mayavi import mlab

print(__doc__)

fig = mlab.figure(size=(1000,550))

brain = Brain("fsaverage", "both", "cortex.patch.flat",
              subjects_dir='/usr/local/freesurfer/subjects',
              figure=fig,background='w')
brain.add_label(label='V1_exvivo',hemi='lh')
brain.add_label(label='V1_exvivo',hemi='rh')

overlay_file = "example_data/lh.sig.nii.gz"
brain.add_overlay(overlay_file,hemi='lh')

cam = fig.scene.camera
cam.zoom(1.85)

mlab.savefig('test.png',figure=fig,magnification=5) # save a high-res figure