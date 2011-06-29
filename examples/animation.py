from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

#these work
brain.animate(['l', 'a'])
# control number of steps
brain.animate(['l', 'm'], 30)
# any path you can think of
brain.animate(['l', 'p', 'm', 'a', 'p', 'a', 'l'], 45)

# full turns
brain.animate(["m"] * 3)

#movies
brain.animate(['l', 'l'], save_movie=True, fname='simple_animation.avi')

# however, rotating out of the axial plane is not allowed
brain.animate(['l', 'd'])
