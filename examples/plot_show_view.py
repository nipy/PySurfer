"""
==============================
Show the different brain views
==============================

Among the views available are lateral, rostral, caudal, frontal etc.

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
brain.show_view('m')
brain.show_view('rostral')
brain.show_view('caudal')
brain.show_view('ve')
brain.show_view('frontal')
brain.show_view('par')
brain.show_view('dor')

###############################################################################
# More advanced parameters
brain.show_view({'distance': 432})
# with great power comes great responsibility
brain.show_view({'azimuth': 135, 'elevation': 79}, roll=107)