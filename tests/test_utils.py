import pytest
import xarray as xr

from xcdat.utils import compare_datasets, str_to_bool


class TestCompareDatasets:
    def test_returns_unique_coord_and_data_var_keys(self):
        ds1 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(name="data_var1"),
            },
            coords={"coord1": xr.DataArray(name="coord1")},
        )
        ds2 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(name="data_var1"),
                "data_var2": xr.DataArray(name="data_var2"),
            },
            coords={
                "coord1": xr.DataArray(name="coord1"),
                "coord2": xr.DataArray(name="coord2"),
            },
        )

        result = compare_datasets(ds1, ds2)
        assert result["unique_coords"] == ["coord2"]
        assert result["unique_data_vars"] == ["data_var2"]

    def test_returns_nonidentical_and_nonequal_coord_and_data_var_keys(self):
        ds1 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[0]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[0]),
            },
        )
        ds2 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[1]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[1]),
            },
        )

        result = compare_datasets(ds1, ds2)
        assert sorted(result["nonidentical_coords"]) == ["coord1", "coord2"]
        assert sorted(result["nonequal_coords"]) == ["coord1", "coord2"]
        assert sorted(result["nonidentical_data_vars"]) == ["data_var1", "data_var2"]
        assert sorted(result["nonequal_data_vars"]) == ["data_var1", "data_var2"]

    def test_returns_no_differences_between_datasets(self):
        ds1 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[0]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[0]),
            },
        )
        ds2 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[0]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[0]),
            },
        )

        result = compare_datasets(ds1, ds2)
        assert result["nonidentical_coords"] == []
        assert result["nonequal_coords"] == []
        assert result["nonidentical_data_vars"] == []
        assert result["nonequal_data_vars"] == []


class TestStrToBool:
    def test_converts_str_to_bool(self):
        result = str_to_bool("True")
        expected = True
        assert result == expected

        result = str_to_bool("False")
        expected = False
        assert result == expected

    def test_raises_error_if_str_is_not_a_python_bool(self):
        with pytest.raises(ValueError):
            str_to_bool(True)  # type: ignore

        with pytest.raises(ValueError):
            str_to_bool(1)  # type: ignore

        with pytest.raises(ValueError):
            str_to_bool("1")
