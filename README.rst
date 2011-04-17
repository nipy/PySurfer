==============
PySurfer
==============
-----------------------------------------
Python Visualization Tools for FreeSurfer 
-----------------------------------------

Purpose
-------
FreeSurfer_ provides its own standalone visualization tools: tkmedit for viewing volume images and tksurfer for viewing surface images. Both of these tools are adequate in their own right, but scripting them is difficult and cumbersome. There’s a need for easier-to-use visualization tools that can be potentially used from existing pipeline tools (NiPype_). Rather than develop a 3D engine from scratch, the Mayavi engine from Enthought (included in EPD_) provides a robust, easy-to-use API for cross-platform visualization.

PySurfer aims to leverage Mayavi's engine and provide most of the visualization capabilities of tksurfer through a simple python interface.

Goals
-----
- A simple, easy to use interface for loading, visualizing, and saving images of FreeSurfer data.
- A pythonic API that resembles tksurfer's TCL interface as much as possible.

.. _EPD: http://www.enthought.com/products/epd.php
.. _FreeSurfer: http://surfer.nmr.mgh.harvard.edu/
.. _NiPype: http://www.nipy.org/nipype
