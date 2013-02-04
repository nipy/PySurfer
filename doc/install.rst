.. _install:

Installation and Dependencies
=============================

Installing PySurfer is quite simple with easy_install_::

    easy_install -U pysurfer

or pip_::

    pip install pysurfer

To use PySurfer, you will need to have the following Python packages:

* numpy_
* scipy_
* ipython_
* nibabel_
* mayavi_
* matplotlib_
* PIL_

An easy option to get all of these packages is to use the Enthought
Python Distribution (EPD_) which is free for academic use.

.. include:: links_names.txt

.. note::

    For optimal results and performance make sure IPython is invoked using the
    best GUI back end. In an EPD environment on Mac OS X this would be WX,
    so `ipython --pylab wx`. On Linux Debian / Ubuntu (Neurodebian) QT seems
    to be more appropriate, so `ipython --pylab qt`. Reports for other platforms
    and distributions are appreciated.
