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
brain.save_montage('/tmp/fsaverage_h_montage.png', ['l', 'v', 'm'], orientation='v')
brain.close()

###############################################################################
# View created image
import Image
import pylab as pl
image = Image.open('/tmp/fsaverage_h_montage.png')
fig = pl.figure(figsize=(5,3))
pl.imshow(image, origin='lower')
pl.xticks(())
pl.yticks(())
