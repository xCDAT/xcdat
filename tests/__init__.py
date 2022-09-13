"""Unit test package for xcdat."""
from xarray.core.options import set_options
from xarray.tests import _importorskip, requires_dask  # noqa: F401

set_options(warn_for_unclosed_files=False)


has_xesmf, requires_xesmf = _importorskip("xesmf")
