"""Top-level package for xcdat."""
from xcdat.bounds import BoundsAccessor  # noqa: F401
from xcdat.dataset import decode_time_units, open_dataset, open_mfdataset  # noqa: F401
from xcdat.spatial_avg import SpatialAverageAccessor  # noqa: F401
from xcdat.xcdat import XCDATAccessor  # noqa: F401

__version__ = "0.1.0"
