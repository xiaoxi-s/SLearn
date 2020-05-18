.. SLearn documentation master file, created by
   sphinx-quickstart on Mon May 18 10:55:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SLearn
======

SLearn is a Machine Learning library implemented with Eigen3, C++. Python interface is 
also provided.


Installation Guide
^^^^^^^^^^^^^^^^^^

1. Install installation tools: git, cmake, gcc
2. Download and install dependencies: Eigen3, pybind11, google test (if want to run test on C++ end), pytest (if want to run test on python end)
3. Download Slean and go to the root directory (where setup.py is) run the following command to install::

    python3 setup.py install

4. You are ready to go!

Guide
^^^^^
Please refer to the document

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   linear_model


.. find files


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
