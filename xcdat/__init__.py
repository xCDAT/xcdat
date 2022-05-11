"""Top-level package for xcdat."""
from xcdat.axis import swap_lon_axis  # noqa: F401
from xcdat.bounds import BoundsAccessor  # noqa: F401
from xcdat.dataset import decode_non_cf_time, open_dataset, open_mfdataset  # noqa: F401
from xcdat.regridder.accessor import RegridderAccessor  # noqa: F401
from xcdat.spatial import SpatialAccessor  # noqa: F401
from xcdat.temporal import TemporalAccessor  # noqa: F401
from xcdat.utils import compare_datasets  # noqa: F401

__version__ = "0.2.0"
