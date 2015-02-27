"""
============================
Animate brain and save movie
============================

"""
from surfer import Brain

print(__doc__)

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

brain.animate(['l', 'c'])

# control number of steps
brain.animate(['l', 'm'], n_steps=30)

# any path you can think of
brain.animate(['l', 'c', 'm', 'r', 'c', 'r', 'l'], n_steps=45)

# full turns
brain.animate(["m"] * 3)

# movies
brain.animate(['l', 'l'], n_steps=10, fname='simple_animation.avi')

# however, rotating out of the axial plane is not allowed
try:
    brain.animate(['l', 'd'])
except ValueError as e:
    print(e)
