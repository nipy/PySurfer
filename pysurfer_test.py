#! /usr/bin/env python
import os
import sys
from os.path import join as pjoin
from enthought.mayavi import mlab
sys.path.append("/cluster/kuperberg/SemPrMM/scripts/PySurfer")
import pysurfer as ps
import random

mlab.options.offscreen = True

f = mlab.figure(random.randint(0,1000), bgcolor=(.93,.90,.734), size=(800,800))
mlab.clf()
f.scene.disable_render = True

data_dir = os.environ["SUBJECTS_DIR"]

sub = "ya24"
hemi = "lh"
sur = "inflated"


surf = ps.Surface(subject=sub, hemi=hemi,surface=sur)
#print "Loading geometry"
#surf.load_geometry(pjoin(data_dir, subject, "surf", "%s.%s"%(hemi,surface)))
#surf.load_curvature(pjoin(data_dir, subject, "surf", "%s.curv"%hemi))
surface_mesh = surf.get_mesh()
brain = mlab.pipeline.surface(surface_mesh, colormap="bone", vmin=-.5, vmax=1.5)
#bar = mlab.scalarbar()
#bar.reverse_lut =True
#bar.visible=False


# stats = ps.Surface()
# print "Loading statistic overlay"
# stats.load_scalar_data(sys.argv[1])
# stats_mesh = mlab.pipeline.triangular_mesh_source(surf.x,
#                                                   surf.y,
#                                                   surf.z,
#                                                   surf.faces,
#                                                   scalars=stats.scalar_data)
#thresh = mlab.pipeline.threshold(stats_mesh, low=2.3)
#stats_surf = mlab.pipeline.surface(thresh,colormap="hot")
#bar = mlab.scalarbar(stats_surf)


print "Writing snapshots"
mlab.view(180,90)
for view in ["lat", "post", "med", "ant"]:
    mlab.draw()
    mlab.savefig("test-%s.png"%view)
    f.scene.camera.azimuth(90)
"""
mlab.savefig("test-lat.png")
mlab.view(270,90)
mlab.savefig("test-post.png")
mlab.view(0,90)
mlab.savefig("test-med.png")
mlab.view(90,90)
mlab.savefig("test-ant.png")
"""
mlab.close(all=True)