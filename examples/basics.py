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
brain.show_view('m')
brain.show_view('rostral')
brain.show_view('caudal')
brain.show_view('dor')
brain.show_view('ve')
brain.show_view('frontal')
brain.show_view('par')

###############################################################################
# save some images
brain.show_view('lat')
brain.save_image("%s_lat.png" % sub)

brain.save_imageset(sub, ['med', 'lat', 'ros', 'caud'], 'jpg')

###############################################################################
# More advanced parameters
brain.show_view({'distance': 375})
# with great power comes great responsibility
brain.show_view({'azimuth': 20, 'elevation': 30}, roll=20)

###############################################################################
# Save a set of images as a montage
brain.save_montage('fsaverage_h_montage.png', ['l', 'v', 'm'], orientation='h')
brain.save_montage('fsaverage_v_montage.png', ['l', 'v', 'm'], orientation='v')
