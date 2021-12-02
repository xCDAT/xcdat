"""Axis module for utilities related to axes."""

from typing import Dict

from typing_extensions import Literal

# Mapping of CF compliant long and short axis keys to their generic
# representations. This map is useful for indexing a Dataset/DataArray on
# a key by falling back on the generic version. Attempting to index on the short
# key when the long key is used will fail, but using the generic key should
# work.
CFAxis = Literal["lat", "latitude", "Y", "lon", "longitude", "X", "time", "T"]
GenericAxis = Literal["X", "Y", "T"]
GENERIC_AXIS_MAP: Dict[CFAxis, GenericAxis] = {
    "lat": "Y",
    "latitude": "Y",
    "Y": "Y",
    "lon": "X",
    "longitude": "X",
    "X": "X",
    "time": "T",
    "T": "T",
}
