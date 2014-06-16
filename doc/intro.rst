.. _intro:

What is PySurfer?
=================

PySurfer is a Python library and application for visualizing brain imaging
data. It is specifically useful for plotting data on a three-dimensional mesh
representing the cortical surface of the brain. If you have functional MRI,
magnetoencephalography, or anatomical measurements from cortex, PySurfer can
help you turn them into beautiful and reproducible graphics.

PySurfer uses an explicit model of cortical geometry to generate
highly-accurate images of neuroimaging data. This is preferable to other
approaches that use simple 3D renderings of a brain volume because the
underlying topology of the cortex is a two-dimensional sheet.  PySurfer can
read cortical models that have been processed using Freesurfer_ to "inflate"
the cortical folds and reveal activations that are buried within deep sulci.
This presentation is much closer to how cortical areas are laid out, and it can
help you understand and communicate your efforts to map functional or
morphometric organization.

PySurfer and its dependencies are written in Python and released with a liberal
open source license. PySurfer can be combined with other tools from the nipy_
ecosystem to manipulate and plot data in the same script or interactive
session. The visualization is primarily controlled with a high-level API that
allow you to draw a complex scene with just a few lines of code. This means
that PySurfer is naturally scriptable. Once you have developed a basic
visualization, it's easy to add a for-loop and generate the same image for
every subject in your experiment. It also aids the reproducibility of graphics
you create for conferences or papers, as each figure can be associated with a
short script that shows exactly how the visualization was produced.

To see a set of examples demonstrating some of PySurfer's capabilities, you can
check out the :ref:`example gallery <examples>`.

.. include:: links_names.txt

