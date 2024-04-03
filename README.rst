========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs| |binder|
    * - tests
      - | |CI| |PyPiPublished| |codecov| |codacy|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
    * - Gitter
      - | General |gitter| Standard |std| Collusion |collusion| OneShot |oneshot|
.. |docs| image:: https://readthedocs.org/projects/scml/badge/?style=flat
    :target: https://scml.readthedocs.io/en/latest
    :alt: Documentation Status

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

.. |commits-since| image:: https://img.shields.io/github/commits-since/yasserfarouk/scml/v0.7.4.svg
    :alt: Commits since latest release
    :target: https://github.com/yasserfarouk/scml/compare/v0.7.4...master

.. |CI| image:: https://github.com/yasserfarouk/scml/workflows/CI/badge.svg
    :target: https://www.github.com/yasserfarouk/scml
    :alt: Build Status

.. |PyPiPublished| image:: https://github.com/yasserfarouk/scml/workflows/PyPI/badge.svg
    :target: https://pypi.python.org/pypi/scml
    :alt: Published on Pypi

.. |gitter| image:: https://badges.gitter.im/scml-anac/community.svg
   :alt: Join the chat at https://gitter.im/scml-anac/community
   :target: https://gitter.im/scml-anac/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |std| image:: https://badges.gitter.im/scml-anac/standard.svg
   :alt: Join the chat at https://gitter.im/scml-anac/standard
   :target: https://gitter.im/scml-anac/standard?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |collusion| image:: https://badges.gitter.im/scml-anac/collusion.svg
   :alt: Join the chat at https://gitter.im/scml-anac/collusion
   :target: https://gitter.im/scml-anac/collusion?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |oneshot| image:: https://badges.gitter.im/scml-anac/ones-hot.svg
   :alt: Join the chat at https://gitter.im/scml-anac/one-shot
   :target: https://gitter.im/scml-anac/one-shot?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :alt: run in binder
   :target: https://mybinder.org/v2/gh/yasserfarouk/scml/HEAD

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


We only support python 3.10 and 3.11. The reason python 3.12 is not yet supported is that stable_baselines3 is
not supporting it yet.


Documentation
=============


https://scml.readthedocs.io


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
