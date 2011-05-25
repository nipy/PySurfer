# from surfer import Brain
# 
# sub = 'fsaverage'
# hemi = 'lh'
# surf = 'inflated'
# 
# brain = Brain(sub, hemi, surf)
# 
# #these work
# brain.animate(['l', 'a'])
# brain.animate(['l', 'm'], 30)
# brain.animate(['a', 'm'], 10)
# brain.animate(['l', 'p', 'm', 'a', 'p', 'm', 'l'], 45)
# 
# # full turns
# brain.animate(["m"] * 3)
# 
# #movies
# brain.animate(['l', 'l'], save_movie=True, fname='output.avi')
# 
# 
# #work, look weird at the end
# brain.animate(['l', 'd'])
# brain.animate(['l', 'v'])
# 
# 
# #weird and difficult to solve
# brain.animate(['p', 'd'])
# brain.animate(['p', 'v'])
# brain.animate(['d', 'd'])

from surfer import viz
from enthought.tvtk.api import tvtk
import numpy as np
from enthought.mayavi import mlab

brain = viz.Brain('bert', 'lh', 'inflated')

def norm(q):
	mag = np.sqrt(sum(np.array(q) ** 2))
	return tuple(np.array(q) / mag)

def v2q(brain, v):
	q = {'dorsal': (0., 0., 1, 0.),
		 'ventral': (0., 0., -1., 0.),
		 'lateral': (0., 1., 0., 0.),
		 'medial': (0., -1., 0., 0.),
		 'anterior': (0., 0., 0., 1.),
		 'ventral': (0., 0., 0., -1.)}
	#gv = brain.xfm_view(v)
	return norm(q[v])	
	
def q2v(brain, q):
	pass

def show_q(brain, q):
	pass

qint = tvtk.QuaternionInterpolator()


f = 'lateral'
l = 'medial'

n = 60

brain.show_view(f)
q1 = brain._f.scene.camera.orientation_wxyz
brain.show_view(l)
q2 = brain._f.scene.camera.orientation_wxyz

qint.add_quaternion(0, q1)
qint.add_quaternion(1, q2)

brain.show_view(f)
for i in np.array(range(1, n)) / float(n):
	out = np.zeros(4)
	qint.interpolate_quaternion(i, out)
	print out
# 	tr = tvtk.Transform()
# 	tr.rotate_wxyz(*out)
# 	brain._f.scene.camera.apply_transform(tr)
	brain._f.scene.camera_
# 	brain._f.scene.render()
