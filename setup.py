#!/usr/bin/env python

"""The setup script."""

from typing import List

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# https://packaging.python.org/discussions/install-requires-vs-requirements/#install-requires
install_requires: List[str] = [
    "cf_xarray",
    "dask",
    "numpy",
    "pandas",
    "xarray",
    "typing_extensions",
]
test_requires = ["pytest>=3"]

setup(
    author="XCDAT developers",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache-2.0 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Xarray Extended with Climate Data Analysis Tools",
    install_requires=install_requires,
    license="Apache-2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="xcdat",
    name="xcdat",
    packages=find_packages(include=["xcdat", "xcdat.*"]),
    test_suite="tests",
    tests_require=test_requires,
    url="https://github.com/XCDAT/xcdat",
    version="0.1.0rc1",
    zip_safe=False,
)
