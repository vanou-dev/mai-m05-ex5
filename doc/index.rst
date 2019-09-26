.. vim: set fileencoding=utf-8 :

.. _logreg_iris:

===============================================================
 Reproducible Multi-Class Logistic Regression for Iris Flowers
===============================================================

.. todolist::

This packakge contains the code I used to generate the tables on the paper:


.. code-block:: bibtex

   @inproceedings{iris2019,
     author = {John Doe},
     title = {A Simple Solution to Iris Flower Classification},
     year = {2019},
     month = jun,
     booktitle = {Reproducible Research Conference, Martigny, 2019},
     url = {http://example.com/path/to/my/article.pdf},
   }


We appreciate your citation in case you use results obtained directly or
indirectly via this software package.


Installation
------------

This package depends on numpy_ and scipy_ to run properly. Please install a
modern version of these packages before trying to run the code examples.

The tests on my paper were executed on a machine running a docker and using the
docker image `fspr/lab:2019
<https://cloud.docker.com/u/fspr/repository/docker/fspr/lab>`_.  installation
with Python 3.6.8.


Running
-------

I created a script that can run the source code reproducing all tables from the
above paper. Run it like so:


.. code-block:: shell

  $ python ./paper.py


The contents of each table in the paper should be printed one after the other.


Troubleshooting
---------------

You can run unit tests I have prepared like this:


.. code-block:: shell

  $ nosetests ./test.py

In case of problems, please get in touch with me `by e-mail
<mailto:john.doe@example.com>`_.


Licensing
---------

This work is licensed under the GPLv3_.


.. Here goes our links
.. include:: links.rst


Technical documentation
-----------------------

.. toctree::

   py_api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
