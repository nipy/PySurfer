import os
import sys
from os.path import join as pjoin

import numpy as np
from enthought.mayavi import mlab

from .io import Surface


class Brain(object):
    
    def __init__(self, subject_id, hemi, surf, curv=True):
        """Initialize a Brain object with Freesurfer-specific data.

        Parameters
        ----------
        subject_id : str
            subject name in Freesurfer subjects dir
        hemi : str
            hemisphere id (ie 'lh' or 'rh')
        surf :  geometry name
            freesurfer surface mesh name (ie 'white', 'inflated', etc.)
        curv : boolean
            if true, loads curv file and displays binary curvature (default: True)
        overlay : filepath
            path to overlay file 
        """
        # Initialize an mlab figure
        self._f = mlab.figure(np.random.randint(1,1000), 
                              bgcolor=(12./256,0./256,25./256), 
                              size=(800,800))
        mlab.clf()
        self._f.scene.disable_render = True

        # Initialize a Surface object as the geometry
        self._geo = Surface(subject_id, hemi, surf, curv=True)

        # Load in the geometry and (maybe) curvature
        self._geo.load_geometry()
        if curv:
            self._geo.load_curvature()
            curv_data = self._geo.bin_curv
        else:
            curv_data = None
    
        # mlab pipeline mesh for geomtery 
        self._geo_mesh = mlab.pipeline.triangular_mesh_source(*self._geo.geometry, 
                                                              scalars=curv_data)

        # mlab surface for the geometry
        colormap, vmin, vmax, reverse = self.__get_geo_colors()
        self._geo_surf = mlab.pipeline.surface(self._geo_mesh, colormap=colormap, vmin=vmin, vmax=vmax)
        if reverse:
            curv_bar = mlab.scalarbar(self._geo_surf)
            curv_bar.reverse_lut = True
            curv_bar.visible = False

        # Initialize the overlay dictionary
        self.overlays = dict()
        
        # Turn disable render off so that it displays
        self._f.scene.disable_render = False

    def add_overlay(self, filepath, range, sign="abs", name=None):
        """Add an overlay to the overlay dict."""

        # If no name provided, get it from the filepath
        if name is None:
            basename = os.path.basename(filepath)
            if basename.endswith(".gz"):
                basename = basename[:-3]
            name = os.path.splitext(basename)[0]
        
        # Check whether an overlay with this name already exists and raise an exception if so
        if name in self.overlays:
            raise NameError(("Overlay with name %s already exists. "
                             "Please provide a name for this overlay"%name))
        


         
    def __get_geo_colors(self):
        """Return an mlab colormap name, vmin, and vmax for binary curvature.

        XXX At the moment just return a default.  Get from the config eventually

        Returns
        -------
        colormap : string
            mlab colormap name
        vmin : float
            curv colormap minimum
        vmax : float
            curv colormap maximum
        reverse : boolean
            boolean indicating whether the colormap should be reverse
            
        """
        return "greys", -1., 2., False


class Overlay(object):
    
