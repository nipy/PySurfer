"""
=======================
Basics of vizualization
=======================

"""
print __doc__

from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

###############################################################################
# show all views
brain.show_view('lateral')
