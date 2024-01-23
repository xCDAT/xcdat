---
title: "xCDAT: A Python Package for Simple and Robust Analysis of Climate Data"
tags:
    - Python
    - python
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
      orcid: 0009-0009-3996-3772
      affiliation: 1
    - name: Jiwoo Lee
      orcid: 0000-0002-0016-7199
      affiliation: 1
    - name: Chengzhu Zhang
      orcid: 0000-0002-9632-0716
      affiliation: 1
affiliations:
    - name: Lawrence Livermore National Lab, Livermore, USA
      index: 1
date: 31 January 2024
bibliography: paper.bib
---

# Summary

xCDAT (Xarray Climate Data Analysis Tools) is an open-source Python package that extends Xarray [@Hoyer:2017] for climate data analysis on structured grids. xCDAT streamlines analysis of climate data by exposing common climate analysis operations through a set of straightforward APIs. Some of xCDAT's key features include spatial averaging, temporal averaging, and regridding. These features are inspired by the Community Data Analysis Tools ([CDAT](https://cdat.llnl.gov/)) library [@Williams:2009] [@Williams:2014] [@cdat] and leverage powerful packages in the [Xarray](https://docs.xarray.dev/en/stable/) ecosystem including [xESMF](https://github.com/pangeo-data/xESMF) [@xesmf], [xgcm](https://xgcm.readthedocs.io/en/latest/) [@xgcm], and [CF xarray](https://cf-xarray.readthedocs.io/en/latest/) [@cf-xarray]. To ensure general compatibility across various climate models, xCDAT operates on datasets that are compliant with the Climate and Forecast (CF) metadata conventions [@Hassell:2017].

# Statement of Need

Analysis of climate data frequently requires a number of core operations, including reading and writing of netCDF files, horizontal and vertical regridding, and spatial and temporal averaging. While many individual software packages address these needs in a variety of computational languages, CDAT stands out because it provides these essential operations via open-source, interoperable Python packages. Since CDAT’s inception, the volume of climate data has grown substantially as a result of both a larger pool of data products and increasing spatiotemporal resolution of model and observational data. Larger data stores are important for advancing geoscientific understanding, but also require increasingly performant software and hardware. These factors have sparked a need for new analysis software that offers core geospatial analysis functionalities capable of efficiently handling large datasets while using modern technologies and standardized software engineering principles.

xCDAT addresses this need by combining the power of Xarray with meticulously developed geospatial analysis features inspired by CDAT. Xarray is the foundation of xCDAT because of its widespread adoption, technological maturity, and ability to handle large datasets with parallel computing via Dask. Xarray is also interoperable with the scientific Python ecosystem (e.g., [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/)), which greatly benefits users who need to use additional analysis tools. Since Xarray is designed as a general-purpose library, xCDAT fills in domain specific gaps by providing features to serve the climate science community _(refer to [Key Features](#key-features))_.

Performance is one fundamental driver in how xCDAT is designed, especially with large datasets. xCDAT conveniently inherits Xarray's support for parallel computing with Dask [@dask:2016]. Parallel computing with Dask enables users to take advantage of compute resources through multithreading or multiprocessing. To use Dask's default multithreading scheduler, users only need to open and chunk datasets in Xarray before calling xCDAT APIs. xCDAT's seamless support for parallel computing enables users to run large-scale computations with minimal effort. If users require more resources, they can also configure and use a local Dask cluster to meet resource-intensive computational needs. Figure 1 shows xCDAT's significant performance advantage over CDAT for global spatial averaging on datasets of varying sizes.

![A performance benchmark for global spatial averaging computations using CDAT (serial only) and xCDAT (serial and parallel with Dask distributed scheduler). xCDAT outperforms CDAT by a wide margin for the 7 GB and 12 GB datasets. Runtimes could not be captured for CDAT with datasets >= 22 GB and xCDAT serial for the 105 GB dataset due to memory allocation errors. The performance benchmark setup and scripts are available in the [`xcdat-validation` repo](https://github.com/xCDAT/xcdat-validation/tree/main/validation/v0.6.0/xcdat-cdat-perf-metrics). __Disclaimer: Performance will vary depending on hardware, dataset shapes/sizes, and how Dask and chunking schemes are configured. There are also some cases where selecting a regional averaging domain (e.g., Niño 3.4) can lead to CDAT outperforming xCDAT.__ \label{fig:figure1}](figures/figure1.png){ height=40% }

xCDAT's intentional design emphasizes software sustainability and reproducible science. It aims to make analysis code reusable, readable, and less-error prone by abstracting common Xarray boilerplate logic into simple and configurable APIs. xCDAT extends Xarray by using [accessor classes](https://docs.xarray.dev/en/stable/internals/extending-xarray.html) that operate directly on Xarray Dataset objects. xCDAT is rigorously tested using real-world datasets and maintains 100% unit test coverage (at the time this paper was written). To demonstrate the value in xCDAT's API design, Figure 2 compares code to calculate annual averages for global climatological anomalies using Xarray against xCDAT. xCDAT requires fewer lines of code and supports further user options (e.g., regional or seasonal averages, not shown). Figure 2 shows the plots for the results produced by xCDAT.

![A comparison of the code to calculate annual averages for global climatological anomalies in A) Xarray and B) xCDAT. xCDAT abstracts most of the Xarray boilerplate logic for calculating weights and grouping data by specific time frequencies, leading to code that is more readable, maintainable, and flexible. The results from both sets of code are within machine precision. \label{fig:figure2}](figures/figure2.png){ height=100% }

![A) Monthly surface skin temperature anomalies for September 1850. B) Monthly (gray) and annual (black) global mean surface skin temperature anomaly values. Temperature data is from an E3SMv2 climate model [@Golaz:2022] simulation over the historical period (1850 – 2014). \label{fig:figure3}](figures/figure3.png){ height=45% }

xCDAT's mission is to provide a maintainable and extensible package that serves the needs of the climate community in the long-term. xCDAT is a community-driven project and the development team encourages all who are interested to get involved through the [GitHub repository](https://github.com/xCDAT/xcdat).

# Key Features

## Extension of `xarray.open_dataset()` and `xarray.open_mfdataset()` with post-processing options

xCDAT extends `xarray.open_dataset()` and `xarray.open_mfdataset()` with additional post-processing operations to support climate data analysis. These APIs can generate missing coordinate bounds for the X, Y, T, and/or Z axes and lazily decode time coordinates represented by `cftime` ([more info](https://github.com/xCDAT/xcdat/pull/489#issuecomment-1579275827)). Other functionality includes re-centering time coordinates between time bounds and converting the longitudinal axis orientation between [0, 360) and [-180, 180).

## Robust interpretation of CF metadata

xCDAT uses [CF xarray](https://cf-xarray.readthedocs.io/en/latest/) to interpret CF metadata present in datasets, enabling xCDAT to operate generally across model and observational datasets that are CF-compliant. This feature enables xCDAT to generate missing coordinate bounds, recognize the coordinates and coordinate bounds to use for computational operations, and lazily decode time coordinates based on the CF calendar attribute.

## Temporal averaging

xCDAT's temporal averaging API utilizes Xarray and Pandas. It includes features for calculating time series averages (single-snapshot), grouped time series averages (e.g., seasonal or annual averages), climatologies, and departures. Averages can be weighted (default) or unweighted. There are optional configurations for seasonal grouping including how to group the month of December (DJF or JFD) and defining custom seasons to group by.

## Geospatial weighted averaging

xCDAT’s geospatial weighted averaging supports rectilinear grids with an option to compute averages over a regional domain (e.g., tropical region, Niño 3.4 region).

## Horizontal structured regridding

xCDAT makes use of [xESMF](https://pangeo-xesmf.readthedocs.io/en/latest/) for horizontal regridding capabilities. It simplifies and extends the xESMF horizontal regridding API by generating missing bounds and ensuring bounds and metadata are preserved in the output dataset. xCDAT also offers a Python implementation of [regrid2](https://cdms.readthedocs.io/en/latest/regrid2.html) for handling cartesian latitude by longitude grids.

## Vertical structured regridding

xCDAT makes use of [xgcm](https://xgcm.readthedocs.io/en/latest/) for vertical regridding capabilities. It simplifies and extends the xgcm vertical regridding API by automatically attempting to determine the grid point position relative to the bounds, transposing the output data to match the dimensional order of the input data, and ensuring bounds and metadata are preserved in the output dataset.

# Documentation & Case Studies

The xCDAT [documentation](https://xcdat.readthedocs.io/en/stable/index.html) includes the [public API list](https://xcdat.readthedocs.io/en/stable/api.html) and a Jupyter Notebook [Gallery](https://xcdat.readthedocs.io/en/stable/gallery.html) that demonstrates real-world applications of xCDAT:

-   [A Gentle Introduction to xCDAT (Xarray Climate Data Analysis Tools)](https://xcdat.readthedocs.io/en/stable/examples/introduction-to-xcdat.html)
-   [General Dataset Utilities](https://xcdat.readthedocs.io/en/stable/examples/general-utilities.html)
-   [Calculate Geospatial Weighted Averages from Monthly Time Series](https://xcdat.readthedocs.io/en/stable/examples/spatial-average.html)
-   [Calculate Time Averages from Time Series Data](https://xcdat.readthedocs.io/en/stable/examples/temporal-average.html)
-   [Calculating Climatology and Departures from Time Series Data](https://xcdat.readthedocs.io/en/stable/examples/climatology-and-departures.html)
-   [Horizontal Regridding](https://xcdat.readthedocs.io/en/stable/examples/regridding-horizontal.html)
-   [Vertical Regridding](https://xcdat.readthedocs.io/en/stable/examples/regridding-vertical.html)

# Distribution

xCDAT is available for Linux, MacOS, and Windows via the conda-forge channel on [Anaconda](https://anaconda.org/conda-forge/xcdat). The [GitHub Repository](https://github.com/xCDAT/xcdat) is where we host all development activity. xCDAT is released under the Apache 2-0 license.

# Projects using xCDAT

xCDAT is actively being integrated as a core component of the [Program for Climate Model Diagnosis and Intercomparison (PCMDI) Metrics Package](https://pcmdi.github.io/pcmdi_metrics/) [@pcmdi-metrics] [@Lee:2023] and the [Energy Exascale Earth System Model (E3SM) Diagnostics Package](https://e3sm-project.github.io/e3sm_diags/_build/html/main/index.html) [@Zhang:2022] [@e3sm-diags]. xCDAT is also included in the [E3SM Unified Anaconda Environment](https://e3sm.org/resources/tools/other-tools/e3sm-unified-environment/) [@e3sm-unified] deployed on numerous U.S Department of Energy supercomputers to run E3SM software tools. Members of the development team are also active users of xCDAT and apply the software to advance their own climate research [@Po-Chedley:2022].

# Acknowledgements

xCDAT is jointly developed by scientists and developers at Lawrence Livermore National Laboratory ([LLNL](https://www.llnl.gov/)) from the Energy Exascale Earth System Model ([E3SM](https://e3sm.org/)) Project and Program for Climate Model Diagnosis and Intercomparison ([PCMDI](https://pcmdi.llnl.gov/)). The work is performed for the E3SM project, which is sponsored by Earth System Model Development ([ESMD](https://climatemodeling.science.energy.gov/program/earth-system-model-development)) program, and the Simplifying ESM Analysis Through Standards ([SEATS](https://www.seatstandards.org/)) project, which is sponsored by the Regional and Global Model Analysis ([RGMA](https://climatemodeling.science.energy.gov/program/regional-global-model-analysis)) program. ESMD and RGMA are programs for the Earth and Environmental Systems Sciences Division ([EESSD](https://science.osti.gov/ber/Research/eessd)) in the Office of Biological and Environmental Research ([BER](https://science.osti.gov/ber)) within the [Department of Energy](https://www.energy.gov/)'s [Office of Science](https://science.osti.gov/). This work is performed under the auspices of the U.S. Department of Energy by LLNL under Contract No. DE-AC52-07NA27344.

Thank you to all of the xCDAT contributors and users including Rob Jacob, Ana Ordonez, Mark Zelinka, Christopher Terai, Min-Seop Ahn, Celine Bonfils, Jean-Yves Peterschmitt, Olivier Marti, Andrew Manaster, and Andrew Friedman. We also give a special thanks to Karl Taylor, Peter Gleckler, Paul Durack, and Chris Golaz who all have provided valuable knowledge and guidance throughout the course of this project.

# References
