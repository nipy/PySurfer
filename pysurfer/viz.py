import os
import sys
from os.path import join as pjoin

import numpy as np

from .io import Surface


class Brain(object):
    """Brain object for visualizing with mlab."""

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
            if true, loads curv file and displays binary curvature
            (default: True)
        overlay : filepath
            path to overlay file
        """
        from enthought.mayavi import mlab

        # Set the identifying info
        self.subject_id = subject_id
        self.hemi = hemi
        self.surf = surf

        # Initialize an mlab figure
        self._f = mlab.figure(np.random.randint(1, 1000),
                              bgcolor=(12. / 256, 0. / 256, 25. / 256),
                              size=(800, 800))
        mlab.clf()
        self._f.scene.disable_render = True

        # Initialize a Surface object as the geometry
        self._geo = Surface(subject_id, hemi, surf)

        # Load in the geometry and (maybe) curvature
        self._geo.load_geometry()
        if curv:
            self._geo.load_curvature()
            curv_data = self._geo.bin_curv
        else:
            curv_data = None

        # mlab pipeline mesh for geomtery
        self._geo_mesh = mlab.pipeline.triangular_mesh_source(
                                        self._geo.x, self._geo.y, self._geo.z,
                                        self._geo.faces, scalars=curv_data)

        # mlab surface for the geometry
        colormap, vmin, vmax, reverse = self.__get_geo_colors()
        self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                    colormap=colormap, vmin=vmin, vmax=vmax)
        if reverse:
            curv_bar = mlab.scalarbar(self._geo_surf)
            curv_bar.reverse_lut = True
            curv_bar.visible = False

        # Initialize the overlay dictionary
        self.overlays = dict()

        # Turn disable render off so that it displays
        self._f.scene.disable_render = False

    def add_overlay(self, filepath, range, sign="abs", name=None):
        """Add an overlay to the overlay dict.

        Parameters
        ----------
        filepath : str
            path to the overlay file (must be readable by Nibabel, or .mgh
        range : (min, max)
            threshold and saturation point for overlay display
        sign : {'abs' | 'pos' | 'neg'}
            whether positive, negative, or both values should be displayed
        name : str
            name for the overlay in the internal dictionary

        """
        if name is None:
            basename = os.path.basename(filepath)
            if basename.endswith(".gz"):
                basename = basename[:-3]
            name = os.path.splitext(basename)[0]

        if name in self.overlays:
            raise NameError("Overlay with name %s already exists. "
                            "Please provide a name for this overlay" % name)

        if not sign in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")

    def __get_geo_colors(self):
        """Return an mlab colormap name, vmin, and vmax for binary curvature.

        At the moment just return a default.  Get from the config eventually

        Returns
        -------
        colormap : string
            mlab colormap name
        vmin : float
            curv colormap minimum
        vmax : float
            curv colormap maximum
        reverse : boolean
            boolean indicating whether the colormap should be reversed

        """
        return "gray", -1., 2., True


class Overlay(object):

    def __init__(self, subject_id, hemi, surf, filepath, range, sign):
        """
        """
        if sign in ["abs", "pos"]:
            self._pos = Surface(subject_id, hemi, surf)
        if sign in ["abs", "neg"]:
            self._neg = Surface(subject_id, hemi, surf)
