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
bgcolor = 'w'

brain = Brain(sub, hemi, surf, config_opts={'background': bgcolor})

###############################################################################
# Get a set of images as a montage, note the data could be saved if desired
image = brain.save_montage(None, ['l', 'v', 'm'], orientation='v')
brain.close()

###############################################################################
# View created image
import pylab as pl
fig = pl.figure(figsize=(5, 3), facecolor=bgcolor)
ax = pl.axes(frameon=False)
ax.imshow(image, origin='upper')
pl.xticks(())
pl.yticks(())
pl.draw()
pl.show()
