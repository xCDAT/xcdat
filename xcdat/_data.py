from pathlib import Path

import pooch

from xcdat._logger import _setup_custom_logger

BASE_URL = "https://raw.githubusercontent.com/xCDAT/xcdat-data/main/resources/"

REGISTRY = {
    "navy_land.nc": "sha256:652dc16af076ee2c407cc627ac4787f365ed5d49ca011daf3ded7a652a8c5fce",
}

logger = _setup_custom_logger(__name__)


def _get_pcmdi_mask_path() -> Path:
    """
    Fetch and cache the canonical PCMDI land/sea mask from xcdat-data.

    This function ensures that the PCMDI land/sea mask is always available by
    downloading it from the xcdat-data repository and caching it locally. The
    cache is managed automatically by pooch.

    Caching behavior:
        - Files are cached in the platform-specific cache directory
          (e.g., ``~/.cache/xcdat`` on Linux/macOS,
          ``%LOCALAPPDATA%\\xcdat\\Cache`` on Windows).
        - The cache location can be overridden by setting the ``XCDAT_DATA_DIR``
          environment variable.
        - Integrity is guaranteed by verifying a SHA256 checksum.

    For offline workflows, you can pre-download the mask with:

    >>> from xcdat._data import _get_pcmdi_mask_path
    >>> path = _get_pcmdi_mask_path()

    Returns
    -------
    Path
        The path to the cached PCMDI land/sea mask file.

    References
    ----------
    - [1] xcdat-data repository: https://github.com/xCDAT/xcdat-data
    """
    fetcher = pooch.create(
        path=pooch.os_cache("xcdat"),
        base_url=BASE_URL,
        registry=REGISTRY,
        env="XCDAT_DATA_DIR",
    )
    filepath = fetcher.fetch("navy_land.nc")

    return Path(filepath)
