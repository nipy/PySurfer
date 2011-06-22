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
import math

brain = viz.Brain('fsaverage', 'lh', 'inflated')


first = 'lateral'
last = 'medial'

n = 60

def from_axis_angle(wxyz):
    w, x, y, z = wxyz
    theta = np.radians(w)
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    r = np.hypot(np.hypot(x, y), z)
    q = np.array((c, s*x/r, s*y/r, s*z/r))
    return q

def as_axis_angle(q):
    c,x,y,z = q
    angle = 2.0 * math.acos(c)
    if c > (1.0 - 1e-14) or c < (-1.0 + 1e-14):
       axis = (1.0, 0.0, 0.0)
    else:
       r = np.hypot(np.hypot(x,y),z)
       axis = (x/r, y/r, z/r)
    return (np.degrees(angle),) + axis

def qmul(q1, q2):
    # Return q1 * q2
    t1, x1, y1, z1 = q1
    t2, x2, y2, z2 = q2
    t = t1*t2 - x1*x2 - y1*y2 - z1*z2
    x = t1*x2 + t2*x1 + y1*z2 - z1*y2
    y = t1*y2 - x1*z2 + t2*y1 + z1*x2 
    z = t1*z2 + x1*y2 - y1*x2 + t2*z1 
    return (t, x, y, z)

def qdiv(qnum, qden):
    # Return qnum * qden^-1
    m2 = (qden * qden).sum()
    qinv = np.array([qden[0], -qden[1], -qden[2], -qden[3]]) / m2
    return qmul(qnum, qinv)

camera = brain._f.scene.camera

brain.show_view(first)
wxyz1 = camera.orientation_wxyz
q1 = from_axis_angle(wxyz1)
brain.show_view(last)
wxyz2 = camera.orientation_wxyz
q2 = from_axis_angle(wxyz2)

dq = qdiv(q2, q1)
qint = tvtk.QuaternionInterpolator()
qint.add_quaternion(0, (1.0, 0.0, 0.0, 0.0))  # Identity
qint.add_quaternion(1, dq)

r = np.linspace(0, 1, n)[1:]
brain.show_view(first)
# Push the current transformation onto the stack so we can keep it for later.
camera.view_transform_object.push()
# And again, because we pop before we push.
camera.view_transform_object.push()

for i in r:
	out = np.zeros(4)
	qint.interpolate_quaternion(i, out)
	tr = tvtk.Transform()
	tr.rotate_wxyz(*as_axis_angle(out))
	# Revert back to the original transform.
	camera.view_transform_object.pop()
	# Apply the new transform. this pushes on the stack
	camera.apply_transform(tr)
	brain._f.scene.renderer.reset_camera_clipping_range()
	brain._f.scene.render()
