#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

test_requires = ["pytest>=3"]

setup(
    author="xCDAT developers",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache-2.0 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Xarray Climate Data Analysis Tools",
    license="Apache-2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="xcdat",
    name="xcdat",
    packages=find_packages(include=["xcdat", "xcdat.*"]),
    test_suite="tests",
    tests_require=test_requires,
    url="https://github.com/xCDAT/xcdat",
    version="0.4.0",
    zip_safe=False,
)
