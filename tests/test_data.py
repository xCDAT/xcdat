import pytest

import xcdat._data as data


class TestGetPcmdiMaskPath:
    def test_get_path_with_monkeypatch(self, tmp_path, monkeypatch):
        """Simulate fetch without hitting network."""
        fake_file = tmp_path / "navy_land.nc"
        fake_file.write_bytes(b"dummy")

        class DummyFetcher:
            def fetch(self, rel):
                return str(fake_file)

        monkeypatch.setattr(data.pooch, "create", lambda **kwargs: DummyFetcher())

        path = data._get_pcmdi_mask_path()

        assert path.exists()
        assert path.read_bytes() == b"dummy"

    @pytest.mark.network
    def test_get_path_with_real_fetch(self, tmp_path, monkeypatch):
        """Test fetching the file from the network."""
        # Override the XCDAT_DATA_DIR to use a temporary directory
        monkeypatch.setenv("XCDAT_DATA_DIR", str(tmp_path))
        path = data._get_pcmdi_mask_path()

        assert path.exists()
        assert path.stat().st_size > 1000

    @pytest.mark.network
    def test_get_path_from_cache(self, tmp_path, monkeypatch):
        """Test fetching the file from the cache."""
        # Override the XCDAT_DATA_DIR to use a temporary directory
        monkeypatch.setenv("XCDAT_DATA_DIR", str(tmp_path))

        # First fetch to ensure the file is downloaded
        path = data._get_pcmdi_mask_path()
        assert path.exists()
        initial_mtime = path.stat().st_mtime

        # Fetch again to ensure it uses the cached file
        cached_path = data._get_pcmdi_mask_path()
        assert cached_path.exists()
        assert cached_path.stat().st_mtime == initial_mtime
