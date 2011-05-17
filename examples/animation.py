from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

#these work
brain.animate(['l', 'p'])
brain.animate(['l', 'm'], 30)
brain.animate(['a', 'm'])
brain.animate(['l', 'p', 'm', 'a', 'p', 'm', 'l'])
# full turns now work
brain.animate(["m"] * 3)

#these need help
brain.animate(['p', 'd'])
brain.animate(['l', 'd'])
brain.animate(['d', 'v'])
