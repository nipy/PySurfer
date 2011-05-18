from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

#these work
brain.animate(['l', 'a'])
brain.animate(['l', 'm'], 30)
brain.animate(['a', 'm'], 10)
brain.animate(['l', 'p', 'm', 'a', 'p', 'm', 'l'], 45)

# full turns
brain.animate(["m"] * 3)

#work, look weird at the end
brain.animate(['l', 'd'])
brain.animate(['l', 'v'])


#weird and difficult to solve
brain.animate(['p', 'd'])
brain.animate(['p', 'v'])
brain.animate(['d', 'd'])