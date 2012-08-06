.. -*- mode: rst -*-

PySurfer
========

PySurfer is a Python module for interacting with a cortical surface
representations of neuroimaging data from Freesurfer. It extends Mayavi's
powerful visualization engine with a high-level interface for working with
MRI and MEG data.

PySurfer offers both a command-line interface designed to broadly replicate
Freesurfer's Tksurfer program as well as a Python library for writing scripts
to efficiently explore complex datasets.

To goal of the project is to facilitate the production of figures that are
both beautiful and scientifically informative.

Important Links
---------------

- Official source code repository: https://github.com/nipy/PySurfer
- Online documentation (stable): http://pysurfer.github.com/
- NITRC page: http://www.nitrc.org/projects/pysurfer
- Freesurfer: http://surfer.nmr.mgh.harvard.edu/
- Mailing list: http://mail.scipy.org/mailman/listinfo/nipy-devel

Install
-------

This packages uses distutils, which is the default way of installing python
modules. To install in your home directory, use::

    python setup.py install --home

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

For information about dependencies, please see the online documentation:
http://pysurfer.github.com/install.html

License
-------

Available under the BSD (3-clause) license.

Testing
-------

You can launch the test suite for the io library using nosetests from the
source folder.

For the visualization module the best way to test is to build the documentation,
which will run the example scripts and automatically generate static image output.
From the source directory::

    cd doc/
    make clean
    make html

The resulting documentation will live at _build/html/index.html, which can
be compared to the online docs.

Either method will work only if you have Freesurfer installed on your
machine with a valid SUBJECTS_DIR folder.
