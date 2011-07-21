"""
======================
Make a multiview image
======================

Make one image from multiple views.

"""
print __doc__

from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

###############################################################################
# Save a set of images as a montage
brain.save_montage('fsaverage_h_montage.png', ['l', 'v', 'm'], orientation='h')
brain.close()

###############################################################################
# View created image
import Image
import pylab as pl
image = Image.open('fsaverage_h_montage.png')
pl.imshow(image, origin='lower')
pl.xticks(())
pl.yticks(())
