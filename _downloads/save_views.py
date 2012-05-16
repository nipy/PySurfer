"""
===================
Save a set of views
===================

Save some views in png files.

"""
print __doc__

from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

###############################################################################
# save 1 image
brain.show_view('lat')
brain.save_image("%s_lat.png" % sub)

###############################################################################
# save some images
brain.save_imageset(sub, ['med', 'lat', 'ros', 'caud'], 'jpg')
