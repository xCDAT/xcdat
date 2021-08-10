"""Bounds module for functions related to coordinate bounds."""
from typing import Optional, Tuple, get_args

import cf_xarray as cfxr  # noqa: F401
import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.logger import setup_custom_logger

logger = setup_custom_logger("root")

Coord = Literal["lat", "latitude", "lon", "longitude", "time"]
#: Tuple of supported coordinates in xCDAT functions and methods.
SUPPORTED_COORDS: Tuple[Coord, ...] = get_args(Coord)


@xr.register_dataset_accessor("bounds")
class DatasetBoundsAccessor:
    """A class to represent the DatasetBoundsAccessor.

    Examples
    ---------
    Import:

    >>> from xcdat import bounds

    Get coordinate bounds if they exist, otherwise return generated bounds:

    >>> ds = xr.open_dataset("file_path")
    >>> lat_bounds = ds.bounds.get_bounds("lat") # or pass "latitude"
    >>> lon_bounds = ds.bounds.get_bounds("lon") # or pass "longitude"
    >>> time_bounds = ds.bounds.get_bounds("time")

    Get coordinate bounds and don't generate if they don't exist:

    >>> ds = xr.open_dataset("file_path")
    >>> # Throws error if no bounds exist
    >>> lat_bounds = ds.bounds.get_bounds("lat", allow_generating=False)
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def get_bounds_for_all_coords(self, allow_generating=True) -> xr.Dataset:
        """Gets existing bounds or generates new ones for supported coordinates.

        Returns
        -------
        xr.Dataset
        """
        for coord in [*self._dataset.coords]:
            if coord in SUPPORTED_COORDS:
                self.get_bounds(coord, allow_generating)

        return self._dataset

    def get_bounds(
        self,
        coord: Coord,
        allow_generating: bool = True,
        width: float = 0.5,
    ) -> Optional[xr.DataArray]:
        """Get bounds for a coordinate.

        If bounds don't exist, they will be generated and set in the dataset.

        Parameters
        ----------
        coord : Coord
            The "lat" or "lon" coordinate.
        allow_generating : bool, optional
            If True, generate bounds if they don't exist. If False, return
            only existing bounds or throw error if they don't exist, by default
            True.
        width : float, optional
            Represents the width of the bounds relative to the position of the
            points if generating bounds, by default 0.5.

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If an incorrect ``coord`` argument is passed.
        ValueError
            If ``allow_generating=False`` and no bounds were found in the dataset.
        """
        if coord not in SUPPORTED_COORDS:
            raise ValueError(
                "Incorrect `coord` argument. Supported coordinates include: Supported "
                f"arguments include: {', '.join(SUPPORTED_COORDS)}."
            )

        try:
            bounds = self._dataset.cf.get_bounds(coord)
        except KeyError:
            bounds = None
            if bounds is None and not allow_generating:
                raise ValueError(
                    f"{coord} bounds were not found in the dataset, bounds must be generated"
                )

            logger.info(
                f"{coord} bounds were not found in the dataset, generating bounds."
            )
            bounds = self._generate_bounds(coord, width)

        return bounds

    def _generate_bounds(
        self,
        coord: Coord,
        width: float = 0.5,
    ) -> xr.DataArray:
        """Generates bounds for a coordinate using its data points.

        Parameters
        ----------
        coord : Coord
            The coordinate.
        width : float, optional
            Width of the bounds relative to the position of the nearest
            points, by default 0.5.

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If coords dimensions does not equal 1.
        ValueError
            If coords are length of <=1.

        Notes
        -----
        Based on [1]_ ``iris.coords._guess_bounds`` and [2]_ ``cf_xarray.accessor.add_bounds``

        References
        ----------

        .. [1] https://scitools-iris.readthedocs.io/en/stable/generated/api/iris/coords.html#iris.coords.AuxCoord.guess_bounds

        .. [2] https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_bounds.html#
        """
        da_coord: xr.DataArray = self._get_coord(coord)

        # Validate coordinate shape and dimensions
        if da_coord.ndim != 1:
            raise ValueError("Cannot generate bounds for multidimensional coordinates.")
        if da_coord.shape[0] <= 1:
            raise ValueError("Cannot generate bounds for a coordinate of length <= 1.")

        # Retrieve coordinate dimension to calculate the diffs between points.
        dim = da_coord.dims[0]
        diffs = da_coord.diff(dim)

        # Add beginning and end points to account for lower and upper bounds.
        diffs = np.insert(diffs, 0, diffs[0])
        diffs = np.append(diffs, diffs[-1])

        # Get lower and upper bounds by using the width relative to nearest point.
        # Transpose both bound arrays into a 2D array.
        lower_bounds = da_coord - diffs[:-1] * width
        upper_bounds = da_coord + diffs[1:] * (1 - width)
        bounds = np.array([lower_bounds, upper_bounds]).transpose()

        # Clip latitude bounds at (-90, 90)
        if (
            da_coord.name in ("lat", "latitude", "grid_latitude")
            and "degree" in da_coord.attrs["units"]
        ):
            if (da_coord >= -90).all() and (da_coord <= 90).all():
                np.clip(bounds, -90, 90, out=bounds)

        # Add coordinate bounds to the dataset
        var_name = f"{coord}_bnds"
        self._dataset[da_coord.name] = da_coord.assign_attrs(bounds=var_name)
        self._dataset[var_name] = xr.DataArray(
            name=var_name,
            data=bounds,
            coords={coord: da_coord},
            dims=[coord, "bnds"],
            attrs={**da_coord.attrs, "is_generated": "True"},
        )
        return self._dataset[var_name]

    def _get_coord(self, coord: Coord) -> xr.DataArray:
        """Get the matching coordinate in the dataset.

        Parameters
        ----------
        coord : Coord
            The coordinate.

        Returns
        -------
        xr.DataArray
            Matching coordinate in the Dataset.

        Raises
        ------
        TypeError
            If no matching coordinate is found in the Dataset.
        """
        try:
            matching_coord = self._dataset.cf[coord]
        except KeyError:
            raise KeyError(f"No matching coordinates for coord: {coord}")

        return matching_coord


@xr.register_dataarray_accessor("bounds")
class DataArrayBoundsAccessor:
    """A class representing the DataArrayBoundsAccessor.

    Examples
    --------
    Import module:

    >>> from xcdat import bounds
    >>> from xcdat.dataset import open_dataset

    Return attribute (refer to ``tas.bounds__dict__`` for attributes):

    >>> tas.bounds.<attribute>

    Copy axis bounds from parent Dataset to data variable:

    >>> ds = open_dataset("file_path") # Auto-generates bounds if missing
    >>> tas = ds["tas"]
    >>> tas.bounds._copy_from_dataset(ds)
    """

    def __init__(self, dataarray: xr.DataArray):
        self._dataarray = dataarray

    def copy_from_parent(self, dataset: xr.Dataset) -> xr.DataArray:
        """Copies coordinate bounds from the parent Dataset to the DataArray.

        In an xarray.Dataset, variables (e.g., "tas") and coordinate bounds
        (e.g., "lat_bnds") are stored in the Dataset's data variables as
        independent DataArrays that have no link between one another [3]_. As a
        result, this creates an issue when you need to reference coordinate
        bounds after extracting a variable to work on it independently.

        This function works around this issue by copying the coordinate bounds
        from the parent Dataset to the DataArray variable.

        Parameters
        ----------
        dataset : xr.Dataset
            The parent Dataset.

        Returns
        -------
        xr.DataArray
            The data variable with bounds coordinates in the list of coordinates.

        Notes
        -----

        .. [3] https://github.com/pydata/xarray/issues/1475

        """
        da = self._dataarray.copy()
        da = self._set_bounds_dim(dataset)

        coords = [*dataset.coords]
        boundless_coords = []
        for coord in coords:
            if coord in SUPPORTED_COORDS:
                try:
                    bounds = dataset.cf.get_bounds(coord)
                    da[bounds.name] = bounds.copy()
                except KeyError:
                    boundless_coords.append(coord)

        if boundless_coords:
            raise ValueError(
                "The dataset is missing bounds for the following coords: "
                f"{', '.join(boundless_coords)}. Pass the dataset to"
                "`xcdat.dataset.open_dataset` to auto-generate missing bounds first"
            )

        self._dataarray = da
        return self._dataarray

    def _set_bounds_dim(self, dataset: xr.Dataset) -> xr.DataArray:
        """Sets the bounds dimension in the DataArray using the parent Dataset.

        The bounds dimension must be set before adding bounds to the DataArray
        coordinates, otherwise the error below will be thrown:
        ``ValueError: cannot add coordinates with new dimensions to a DataArray``.

        Parameters
        ----------
        dataset : xr.Dataset
            The parent Dataset.

        Returns
        -------
        xr.DataArray
            The data variable with a bounds dimension.

        Raises
        ------
        KeyError
            When no bounds dimension exists in the parent Dataset.
        """
        da = self._dataarray.copy()
        dims = dataset.dims.keys()

        if "bnds" in dims:
            bounds_dim = "bnds"
        elif "bounds" in dims:
            bounds_dim = "bounds"
        else:
            raise KeyError(
                "No bounds dimension in the parent dataset. This indicates that there "
                "are probably no coordinate bounds in the dataset. Pass the "
                "dataset to `xcdat.dataset.open_dataset` to auto-generate bounds."
            )

        da = da.expand_dims(dim={bounds_dim: np.array([0, 1])})

        self._dataarray = da
        return self._dataarray
