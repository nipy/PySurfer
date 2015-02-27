"""
Basic Visualization
===================

Initialize a basic visualization session.

"""
from surfer import Brain

print(__doc__)

"""
Define the three important variables.
Note that these are the first three positional arguments
in tksurfer (and pysurfer for that matter).
"""
subject_id = 'fsaverage'
hemi = 'lh'
surface = 'inflated'

"""
Call the Brain object constructor with these
parameters to initialize the visualization session.
"""
brain = Brain(subject_id, hemi, surface)
