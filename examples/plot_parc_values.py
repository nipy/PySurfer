"""
==================
Display ROI Values
==================

Here we demonstrate how to take the results of an ROI analysis performed within
each region of some parcellation and display those values on the surface to
quickly summarize the analysis.

"""
print(__doc__)

import os
import numpy as np
import nibabel as nib
from surfer import Brain

subject_id = "fsaverage"
hemi = "lh"
surface = "inflated"

"""
Bring up the visualization.
"""
brain = Brain(subject_id, hemi, surface,
              config_opts=dict(background="white"))

"""
Read in the Buckner resting state network annotation. (This requires a
relatively recent version of Freesurfer, or it can be downloaded separately).
"""
aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                          subject_id, "label",
                          hemi + ".Yeo2011_17Networks_N1000.annot")
labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

"""
Make a random vector of scalar data corresponding to a value for each region in
the parcellation.

"""
rs = np.random.RandomState(4)
roi_data = rs.uniform(.5, .75, size=len(names))

"""
Make a vector containing the data point at each vertex.
"""
vtx_data = roi_data[labels]

"""
Display these values on the brain. Use a sequential colormap (assuming
these data move from low to high values), and add an alpha channel so the
underlying anatomy is visible.
"""
brain.add_data(vtx_data, .5, .75, colormap="GnBu", alpha=.8)
