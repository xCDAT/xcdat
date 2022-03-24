"""Top-level package for xcdat."""
from xcdat.axis import swap_lon_axis  # noqa: F401
from xcdat.bounds import BoundsAccessor  # noqa: F401
from xcdat.dataset import (  # noqa: F401
    decode_non_cf_time,
    has_cf_compliant_time,
    open_dataset,
    open_mfdataset,
)
from xcdat.spatial import SpatialAccessor  # noqa: F401
from xcdat.temporal import TemporalAccessor  # noqa: F401
from xcdat.utils import compare_datasets  # noqa: F401
from xcdat.xcdat import XCDATAccessor  # noqa: F401

__version__ = "0.2.0"
