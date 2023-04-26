---
title: "xCDAT: A Python Package for Simple and Robust Analysis of Climate Data"
tags:
    - Python
    - xarray
    - climate science
    - climate research
    - climate data
    - climate data analysis
authors:
    - name: Tom Vo
      orcid: 0000-0002-2461-0191
      affiliation: 1
    - name: Stephen Po-Chedley
      orcid: 0000-0002-0390-238X
      affiliation: 1
    - name: Jason Boutte
      affiliation: 1
      # TODO: Does Jason have an ORCID?
    - name: Jiwoo Lee
      orcid: 0000-0002-0016-7199
      affiliation: 1
    - name: Chengzhu Zhang
      orcid: 0000-0002-9632-0716
      affiliation: 1
affiliations:
    - name: Lawrence Livermore National Lab, Livermore, USA
      index: 1
date: 24 April 2023
bibliography: paper.bib
---

# Summary

xCDAT (Xarray Climate Data Analysis Tools) is an extension of xarray for climate data
analysis on structured grids. It serves as a modern successor to the Community Data
Analysis Tools (CDAT) library. The goal of xCDAT is to provide generalizable climate domain features and utilities with xarray for simple and robust analysis of climate data.

xCDAT's key features includes spatial averaging, temporal averaging, horizontal
regridding, and vertical regridding. Some features are inspired by CDAT, while others
leverage powerful libraries in the xarray ecosystem (e.g., xESMF, cf_xarray) to deliver
simple and robust APIs.

# Statement of need

CDAT Driving Need

-   The CDAT library has provided over 20 years of robust and comprehensive climate data
    analysis and visualization packages for the open-source community. Many scientists
    and software libraries continue to utilize CDAT as a major dependency in their
    workflows. As software technologies advance and the size of data grows, there is a
    driving need for a modern successor to CDAT that is simple to use, performant, and
    robust.

Origins of xCDAT

    In early 2021, a team of scientists and software engineers at LLNL spent several
    months researching for a viable successor to CDAT. They found that libraries such as
    xarray and Iris offered some similar features to CDAT, but these features were not
    exactly comparable to those in CDAT. For example, CDAT implements unique logic for
    handling axis coordinate metadata such as generating coordinate bounds which are
    used in weighted averaging operations.

    The team decided that it was sensible to develop xCDAT as CDAT's successor.
    Xarray was chosen as the core technology because of its stability, maturity,
    extensibility, and interoperability with the SciPy stack (e.g., dask, matplotlib,
    pandas). xCDAT focuses on streamlining the user experience of developing analysis
    code to reduce the complexity in achieving of certain domain-specific features in
    xarray. Below is a code example for calculating the spatial average of tas
    using xarray vs. xcdat. # TODO: Show figure of code example for spatial averaging

    xCDAT aims to be a generalizable package that is compatible with structured grids
    that are CF-compliant (e.g,. CMIP6).

Xarray design

-   https://docs.xarray.dev/en/stable/internals/extending-xarray.html
    "Xarray is designed as a general purpose library and hence tries to avoid including overly domain specific functionality. But inevitably, the need for more domain specific logic arises."

Leveraging Xarray with Dask

# Features

# Documentation

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Karl Taylor, Peter Gleckler, Paul Durack, and
Chris Golaz who all have provided insightful knowledge and guidance in the planning
and implementation of this software.

xCDAT is jointly developed by scientists and developers from the Energy Exascale Earth
System Model (E3SM) Project and Program for Climate Model Diagnosis and Intercomparison
(PCMDI). The work is performed for the E3SM project, which is sponsored by Earth System
Model Development (ESMD) program, and the Simplifying ESM Analysis Through Standards
(SEATS) project, which is sponsored by the Regional and Global Model Analysis (RGMA)
program. ESMD and RGMA are programs for the Earth and Environmental Systems Sciences
Division (EESSD) in the Office of Biological and Environmental Research (BER) within the
Department of Energy's Office of Science.

# References

Xarray
CDAT
