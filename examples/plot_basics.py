"""
Basic Visualization
===================

Initialize a basic visualization session.

"""
print __doc__

from surfer import Brain

subject_id = 'fsaverage'
hemi = 'lh'
surface = 'inflated'

brain = Brain(subject_id, hemi, surface)
