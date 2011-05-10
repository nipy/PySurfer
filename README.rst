.. -*- mode: rst -*-

About
=====

PySurfer is a python module for interacting with a FreeSurfer data
base and visualizing data using Mayavi2.

Available under the BSD (3-clause) license.

Install
=======

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install

Testing
-------

You can launch the test suite using nosetests from the source folder.

It will only work if you have Freesurfer installed on your machine
with a valid SUBJECTS_DIR folder.
