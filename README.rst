========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |CI| |PyPiPublished| |requires|
        | |codecov|
        | |codacy|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/scml/badge/?style=flat
    :target: http://www.yasserm.com/scml/scml2020docs
    :alt: Documentation Status

.. |requires| image:: https://requires.io/github/yasserfarouk/scml/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/yasserfarouk/scml/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/yasserfarouk/scml/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/yasserfarouk/scml
    :alt: Coverage Status

.. |codacy| image:: https://img.shields.io/codacy/grade/f9512287d5d0485a80cf39e75dfc6d22.svg
    :target: https://www.codacy.com/app/yasserfarouk/scml
    :alt: Codacy Code Quality Status

.. |version| image:: https://img.shields.io/pypi/v/scml.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/scml

.. |wheel| image:: https://img.shields.io/pypi/wheel/scml.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/scml

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/scml.svg
    :alt: Supported versions
    :target: https://pypi.org/project/scml

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/scml.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/scml

.. |commits-since| image:: https://img.shields.io/github/commits-since/yasserfarouk/scml/v0.2.14.svg
    :alt: Commits since latest release
    :target: https://github.com/yasserfarouk/scml/compare/v0.2.14...master

.. |CI| image:: https://github.com/yasserfarouk/scml/workflows/CI/badge.svg
    :target: https://www.github.com/yasserfarouk/scml
    :alt: Build Status

.. |PyPiPublished| image:: https://github.com/yasserfarouk/scml/workflows/PyPI/badge.svg
    :target: https://pypi.python.org/pypi/scml
    :alt: Published on Pypi


.. end-badges

ANAC Supply Chain Management League Platform

Overview
========

This repository is the official platform for running ANAC Supply Chain Management Leagues. It will contain a package
called `scmlXXXX` for the competition run in year XXXX. For example scml2019 will contain all files related to the
2019's version of the competition.


Installation
============

::

    pip install scml

You can also install the in-development version with::

    pip install https://github.com/yasserfarouk/scml/archive/master.zip


Documentation
=============


http://www.yasserm.com/scml/scml2020docs/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
