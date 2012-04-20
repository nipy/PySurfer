"""
Display Volume Data
===================

PySurfer provides a function to sample a volume-encoded file
to the surface for improved visualization. At the moment, this
uses Freesurfer's mri_vol2surf routine.

"""
print __doc__

from surfer import Brain, io

# Bring up the visualization
brain = Brain("fsaverage", "lh", "inflated")

# Project the volume file and return as an array
mri_file = "auto_examples/data/zstat.nii.gz"
reg_file = "auto_examples/data/register.dat"
surf_data = io.project_volume_data(mri_file, "lh", reg_file)

# You can pass this array to the add_overlay method for
# a typical activation overlay (with thresholding, etc.)
brain.add_overlay(surf_data, min=2.3, max=3.09, name="zstat")

# You can also pass it to add_data
brain.overlays["zstat"].remove()
brain.add_data(surf_data, min=-4.265, max=4.265)
