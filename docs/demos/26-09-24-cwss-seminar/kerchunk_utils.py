"""Shared utilities for the CWSS remote-Kerchunk demonstration."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import fsspec
import numpy as np
import xarray as xr


KERCHUNK_SITE_URLS = {
    "nersc": "https://g-eba899.6b7bd8.0ec8.data.globus.org/kerchunk",
    "ornl": "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/kerchunk",
}


def find_json_files(
    catalog: Mapping[str, Mapping[str, Any]], **facets: Any
) -> dict[str, Mapping[str, Any]]:
    """Return catalog entries whose facets match every supplied value.

    Parameters
    ----------
    catalog : Mapping[str, Mapping[str, Any]]
        Kerchunk catalog entries keyed by dataset path.
    **facets : Any
        Metadata facet names and required values.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Matching catalog entries keyed by dataset path.
    """
    return {
        path: metadata
        for path, metadata in catalog.items()
        if all(metadata.get(facet) == value for facet, value in facets.items())
    }


def open_kerchunk_reference(
    reference_file: str, site: Literal["nersc", "ornl"] = "ornl"
) -> xr.Dataset:
    """Open a remotely hosted Kerchunk reference as an xarray dataset.

    Parameters
    ----------
    reference_file : str
        Path to the reference JSON file relative to the selected Kerchunk site.
    site : {"nersc", "ornl"}, default: "ornl"
        Kerchunk reference catalog to use.

    Returns
    -------
    xarray.Dataset
        Lazily opened dataset represented by the reference JSON.

    Raises
    ------
    ValueError
        If ``site`` is not a supported reference catalog.
    """
    try:
        reference_url = f"{KERCHUNK_SITE_URLS[site]}/{reference_file}"
    except KeyError as error:
        supported_sites = ", ".join(KERCHUNK_SITE_URLS)
        raise ValueError(
            f"Unsupported Kerchunk site {site!r}. Choose one of: {supported_sites}."
        ) from error

    filesystem = fsspec.filesystem(
        "reference",
        fo=reference_url,
        remote_options={"asynchronous": True},
        remote_protocol="https",
        asynchronous=True,
    )

    return xr.open_dataset(filesystem.get_mapper(""), engine="zarr", consolidated=False)


def calculate_model_trends(
    dataset: xr.Dataset,
    variable_id: str,
    target_grid: xr.Dataset,
    expected_year_count: int,
) -> tuple[float, xr.DataArray]:
    """Calculate global and grid-cell temperature trends for one model.

    The input dataset must first be subset to and loaded for the analysis period.
    The xCDAT accessors are registered when ``xcdat`` is imported by the notebook.

    Parameters
    ----------
    dataset : xarray.Dataset
        Loaded, time-subset source dataset.
    variable_id : str
        Variable used to calculate temperature trends.
    target_grid : xarray.Dataset
        Target grid used for horizontal regridding.
    expected_year_count : int
        Required count of annual samples after time averaging.

    Returns
    -------
    tuple[float, xarray.DataArray]
        Global and grid-cell trends in K decade-1. The global trend is calculated
        from the native-grid annual data, whereas the grid-cell trend is calculated
        after regridding to ``target_grid``.

    Raises
    ------
    ValueError
        If annual averaging does not produce the expected number of samples.
    """
    dataset = dataset.bounds.add_missing_bounds()
    if "height" in dataset.coords:
        dataset = dataset.drop_vars("height")

    annual_dataset = dataset.temporal.group_average(variable_id, freq="year")
    regridded_dataset = annual_dataset.regridder.horizontal(
        variable_id, target_grid, tool="regrid2"
    )
    decimal_year = regridded_dataset.time.dt.decimal_year
    if len(decimal_year) != expected_year_count:
        raise ValueError(
            "Annual time coordinate has "
            f"{len(decimal_year)} samples; expected {expected_year_count}."
        )

    regridded_dataset = regridded_dataset.assign_coords(time=decimal_year)
    trend_dataset = regridded_dataset[variable_id].polyfit(dim="time", deg=1)
    grid_cell_trend = trend_dataset.polyfit_coefficients.sel(degree=1) * 10

    global_mean = annual_dataset.spatial.average(variable_id)[variable_id]
    global_trend, _ = np.polyfit(decimal_year, global_mean.values, 1)

    return float(global_trend * 10), grid_cell_trend


def select_available_models(
    catalog: Mapping[str, Mapping[str, Any]],
    ecs_data: Mapping[str, float],
    excluded_models: Sequence[str] = (),
) -> list[str]:
    """Return sorted catalog models that have ECS values and are not excluded.

    Parameters
    ----------
    catalog : Mapping[str, Mapping[str, Any]]
        Previously filtered Kerchunk catalog entries.
    ecs_data : Mapping[str, float]
        Model equilibrium climate sensitivity values.
    excluded_models : Sequence[str], optional
        Models omitted because their data cannot support this demonstration.

    Returns
    -------
    list[str]
        Sorted model names available to the analysis.
    """
    excluded_model_set = set(excluded_models)
    catalog_models = {metadata["model"] for metadata in catalog.values()}

    return sorted((catalog_models & set(ecs_data)) - excluded_model_set)
