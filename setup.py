#!/usr/bin/env python

import re
from glob import glob
from os.path import basename, dirname, join, splitext

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="scml",
    version="0.7.4",
    description="ANAC Supply Chain Management League Platform",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    long_description_content_type="text/x-rst",  # Optional (see note above),
    author="Yasser Mohammad",
    author_email="yasserfarouk@gmail.com",
    url="https://github.com/yasserfarouk/scml",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        # 'Topic :: Utilities',
    ],
    project_urls={
        "Documentation": "https://scml.readthedocs.io/",
        "Changelog": "https://scml.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/yasserfarouk/scml/issues",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=3.10",
    install_requires=[
        "click",
        "pytest",
        "hypothesis",
        "negmas>=0.10.20",
        "joblib",
        "jupyter",
        "gif",
        "gymnasium",
        # "python-constraint",
        # "prettytable",
        # "pulp",
        # "mip",
        # "stable-baselines3",
    ],
    extras_require={
        "gui": ["pyqt5"],
    },
    setup_requires=["pytest-runner"],
    entry_points={
        "console_scripts": [
            "scml = scml.cli:main",
            "cliadv = scml.cliadv:cli",
        ]
    },
)
