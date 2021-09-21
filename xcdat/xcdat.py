"""Main xcdat module."""
from typing import Dict, Optional

import xarray as xr

from xcdat.bounds import Coord, DatasetBoundsAccessor  # noqa: F401


@xr.register_dataset_accessor("xcdat")
class XCDATAccessor:
    """
    A class representing XCDATAccessor, which combines all of the public methods
    of XCDAT accessors into a centralized namespace ("xcdat").

    Examples
    ========
    Import xcdat module:

    >>> import xcdat

    Access ``XCDATAccessor`` methods:

    >>> ds.xcdat.<name_of_method>
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    @property
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a mapping of coordinate and axis keys to their bounds.

        The dictionary provides all valid CF compliant keys for a coordinate.
        For example, latitude will includes keys for "lat", "latitude", and "Y".

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping coordinate keys to their bounds.
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.bounds

    def fill_missing_bounds(self) -> xr.Dataset:
        """Fills any missing bounds for supported coordinates in the Dataset.

        Returns
        -------
        xr.Dataset
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.fill_missing()

    def get_bounds(self, coord: Coord) -> xr.DataArray:
        """Get bounds for a coordinate.

        Parameters
        ----------
        coord : Coord
            The coordinate key.

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If an incorrect ``coord`` argument is passed.

        ValueError
            If bounds were not found. They must be added.
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.get_bounds(coord)

    def add_bounds(self, coord: Coord, width: float = 0.5) -> xr.Dataset:
        """Add bounds for a coordinate using its data points.

        If bounds already exist, they must be dropped first.

        Parameters
        ----------
        coord : Coord
            The coordinate key.
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If bounds already exist. They must be dropped first.
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.add_bounds(coord, width)
