Getting Started
===============

Set-Up
++++++

The MATS code was orginally developed using the `Enthought Canopy python package distribution <https://www.enthought.com/product/canopy/>`_.  However, further development will use `Anaconda <https://www.anaconda.com/>`_. Example scripts have typically been written and run using  `jupyter notebooks <https://jupyter.org/>`_ (run in jupyter lab), as this allows for code to be run in segments and for iteration on certain code segments.  Any python package distribution should work with the code as long as the dependent packages are installed.


Main Packages
+++++++++++++

* `MATS <https://github.com/usnistgov/MATS>`_

* `HITRAN Application Programming Interface (HAPI) <https://hitran.org/hapi/>`_   The version used in development (v1.1.0.9.6) is available in the MATS repository.


Dependent Packages
++++++++++++++++++
MATS is not written as a basic package, meaning that there are several dependent packages that need to be installed.

If a desired package is not installed then the following command will install it.  Many python package distributions have integrated package managers and required packages can also be installed through that mechanism.

.. code-block:: bash

	 pip install package_name

There is commonly a delay in the most recent package releases available in python package distribution package managers compared to that available through pip install.  The following command line script will update to the newest release if a package is already installed.  This should only be necessary if there is an error when running MATS with the currently installed version.

.. code-block:: bash

	 python -m pip install --upgrade package_name --user

Below are a list of the packages used in MATS.

* `numpy <https://www.numpy.org/>`_ - python's fundamental scientific computing package
* `pandas <https://pandas.pydata.org/>`_ - python data structure package
* `qgrid <https://github.com/quantopian/qgrid>`_ - provides interactive sorting, filtering, and editing of pandas dataframes in jupyter notebooks.  Make sure to follow the install instructions for both installation and jupyter installation or this won't work.  This is an optional package that allows for the :py:class:`MATS.Edit_Fit_Param_Files` to function.
* os, sys - system variables
* `lmfit <https://lmfit.github.io/lmfit-py/fitting.html>`_ - non-linear least-squares minimization and curve-fitting for python
* `matplotlib <https://matplotlib.org/>`_ - python plotting
* `seaborn <https://seaborn.pydata.org/>`_ - pretty plotting
* `scipy.fftpack <https://docs.scipy.org/doc/scipy/reference/fftpack.html>`_ - provides fft functionality
* `jupyter lab <https://jupyterlab.readthedocs.io/en/stable/>`_ - web-based user interface for Project Jupyter
