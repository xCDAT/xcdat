from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

import xcdat as xc
from xcdat.validation import (
    DatasetValidationError,
    ValidationIssue,
    ValidationResult,
    _IssueCollector,
    _validate_bounds,
    validate_dataset,
)


def _create_rectilinear_dataset(with_bounds: bool = True) -> xr.Dataset:
    lat_attrs = {
        "axis": "Y",
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    lon_attrs = {
        "axis": "X",
        "standard_name": "longitude",
        "units": "degrees_east",
    }
    if with_bounds:
        lat_attrs["bounds"] = "lat_bnds"
        lon_attrs["bounds"] = "lon_bnds"

    ds = xr.Dataset(
        {"tas": (("lat", "lon"), np.ones((2, 3)))},
        coords={
            "lat": xr.DataArray([-30.0, 30.0], dims="lat", attrs=lat_attrs),
            "lon": xr.DataArray([0.0, 120.0, 240.0], dims="lon", attrs=lon_attrs),
        },
    )
    if with_bounds:
        ds["lat_bnds"] = xr.DataArray([[-60.0, 0.0], [0.0, 60.0]], dims=("lat", "bnds"))
        ds["lon_bnds"] = xr.DataArray(
            [[-60.0, 60.0], [60.0, 180.0], [180.0, 300.0]],
            dims=("lon", "bnds"),
        )

    return ds


class TestValidationResult:
    def test_returns_structured_empty_result_for_valid_dataset(self):
        result = validate_dataset(_create_rectilinear_dataset())

        assert result == ValidationResult(())
        assert result.errors == ()
        assert result.warnings == ()
        assert result.is_valid
        result.raise_for_errors()

    def test_warning_only_result_is_valid(self):
        result = validate_dataset(_create_rectilinear_dataset(with_bounds=False))

        assert result.errors == ()
        assert len(result.warnings) == 2
        assert result.is_valid
        result.raise_for_errors()

    def test_raise_for_errors_aggregates_error_diagnostics(self):
        issues = (
            ValidationIssue(
                code="first-error",
                severity="error",
                variable="lat",
                problem="Latitude metadata is invalid.",
                operations=("spatial_average",),
                suggestion="Correct the metadata.",
            ),
            ValidationIssue(
                code="second-error",
                severity="error",
                variable="lon",
                problem="Longitude metadata is invalid.",
                operations=("horizontal_regrid",),
                suggestion="Correct the metadata.",
            ),
        )
        result = ValidationResult(issues)

        with pytest.raises(DatasetValidationError) as exc_info:
            result.raise_for_errors()

        assert exc_info.value.issues == issues
        assert "[first-error] lat" in str(exc_info.value)
        assert "[second-error] lon" in str(exc_info.value)

    def test_orders_issues_deterministically(self):
        result = validate_dataset(_create_rectilinear_dataset(with_bounds=False))

        issue_keys = [(issue.code, issue.variable) for issue in result.issues]
        assert issue_keys == sorted(issue_keys)


def test_issue_collector_merges_affected_operations_for_duplicate_issue():
    collector = _IssueCollector()
    collector.add(
        "duplicate",
        "warning",
        "lat",
        "Duplicate problem.",
        ("spatial_average",),
        "Correct the metadata.",
    )
    collector.add(
        "duplicate",
        "warning",
        "lat",
        "Duplicate problem.",
        ("horizontal_regrid",),
        "Correct the metadata.",
    )

    assert collector.result().issues[0].operations == (
        "spatial_average",
        "horizontal_regrid",
    )


def test_rejects_non_dataset():
    with pytest.raises(TypeError, match="xarray.Dataset"):
        validate_dataset(xr.DataArray([1]))  # type: ignore[arg-type]


class TestAxisValidation:
    def test_warns_for_common_coordinate_name_without_cf_metadata(self):
        ds = _create_rectilinear_dataset()
        ds.lon.attrs = {"bounds": "lon_bnds"}

        result = validate_dataset(ds)

        issue = next(
            issue for issue in result.issues if issue.code == "missing-cf-axis-metadata"
        )
        assert issue.variable == "lon"
        assert issue.severity == "warning"
        assert issue.operations == ("spatial_average", "horizontal_regrid")

    def test_errors_for_conflicting_cf_axis_metadata(self):
        ds = xr.Dataset(
            {"var": ("coord", [1.0, 2.0])},
            coords={
                "coord": xr.DataArray(
                    [0.0, 1.0],
                    dims="coord",
                    attrs={"axis": "X", "standard_name": "latitude"},
                )
            },
        )

        result = validate_dataset(ds)

        assert any(
            issue.code == "conflicting-axis-metadata" and issue.variable == "coord"
            for issue in result.errors
        )

    def test_errors_for_name_fallback_conflict(self):
        ds = xr.Dataset(
            {"tas": ("lat", [1.0, 2.0])},
            coords={
                "lat": xr.DataArray(
                    [-30.0, 30.0],
                    dims="lat",
                    attrs={"axis": "X"},
                )
            },
        )

        result = validate_dataset(ds)

        assert any(
            issue.code == "conflicting-axis-metadata" and issue.variable == "lat"
            for issue in result.errors
        )

    def test_errors_for_multiple_dimension_coordinates_on_one_axis(self):
        ds = xr.Dataset(
            {"var": (("lat", "tau"), np.ones((2, 3)))},
            coords={
                "lat": xr.DataArray([-30.0, 30.0], dims="lat", attrs={"axis": "Y"}),
                "tau": xr.DataArray([0.0, 0.5, 1.0], dims="tau", attrs={"axis": "Y"}),
            },
        )

        result = validate_dataset(ds)

        issue = next(issue for issue in result.errors if issue.code == "ambiguous-axis")
        assert issue.variable == "var"

    def test_reports_multiple_ambiguous_axes_for_one_variable(self):
        ds = xr.Dataset(
            {"var": (("x1", "x2", "y1", "y2"), np.ones((2, 2, 2, 2)))},
            coords={
                "x1": xr.DataArray([0.0, 1.0], dims="x1", attrs={"axis": "X"}),
                "x2": xr.DataArray([0.0, 1.0], dims="x2", attrs={"axis": "X"}),
                "y1": xr.DataArray([0.0, 1.0], dims="y1", attrs={"axis": "Y"}),
                "y2": xr.DataArray([0.0, 1.0], dims="y2", attrs={"axis": "Y"}),
            },
        )

        result = validate_dataset(ds)

        issues = [
            issue
            for issue in result.errors
            if issue.code == "ambiguous-axis" and issue.variable == "var"
        ]
        assert len(issues) == 2
        assert "'X' axis" in issues[0].problem
        assert "'Y' axis" in issues[1].problem


class TestBoundsValidation:
    def test_reports_missing_bounds_as_warning(self):
        result = validate_dataset(_create_rectilinear_dataset(with_bounds=False))

        issues = [issue for issue in result.issues if issue.code == "missing-bounds"]
        assert {issue.variable for issue in issues} == {"lat", "lon"}
        assert all(issue.severity == "warning" for issue in issues)

    def test_errors_if_bounds_attribute_references_missing_variable(self):
        ds = _create_rectilinear_dataset().drop_vars("lat_bnds")

        result = validate_dataset(ds)

        assert any(
            issue.code == "missing-bounds-variable" and issue.variable == "lat"
            for issue in result.errors
        )

    def test_errors_if_bounds_omit_coordinate_dimensions(self):
        ds = _create_rectilinear_dataset().drop_vars("lat_bnds")
        ds["lat_bnds"] = xr.DataArray(np.ones((2, 2)), dims=("other", "bnds"))

        result = validate_dataset(ds)

        assert any(
            issue.code == "bounds-missing-coordinate-dimensions"
            and issue.variable == "lat_bnds"
            for issue in result.errors
        )

    def test_errors_if_bounds_dimension_size_differs_from_coordinate(self):
        coord = xr.DataArray(
            [-30.0, 30.0],
            dims="lat",
            attrs={"bounds": "lat_bnds"},
        )
        bounds = xr.DataArray(np.ones((3, 2)), dims=("lat", "bnds"))
        ds = MagicMock()
        ds.__getitem__.side_effect = {"lat": coord, "lat_bnds": bounds}.__getitem__
        ds.variables = {"lat", "lat_bnds"}
        collector = _IssueCollector()
        axis_coords = {
            "X": set(),
            "Y": {"lat"},
            "T": set(),
            "Z": set(),
        }

        _validate_bounds(ds, axis_coords, collector)  # type: ignore[arg-type]

        assert any(
            issue.code == "bounds-dimension-size-mismatch"
            and issue.variable == "lat_bnds"
            for issue in collector.result().errors
        )

    def test_errors_for_unrelated_bounds_dimensions(self):
        ds = _create_rectilinear_dataset().drop_vars("lat_bnds")
        ds["lat_bnds"] = xr.DataArray(np.ones((2, 3, 2)), dims=("lat", "time", "bnds"))

        result = validate_dataset(ds)

        assert any(
            issue.code == "malformed-bounds-dimensions" and issue.variable == "lat_bnds"
            for issue in result.errors
        )

    def test_errors_for_invalid_bounds_vertex_count(self):
        ds = _create_rectilinear_dataset().drop_vars("lat_bnds")
        ds["lat_bnds"] = xr.DataArray(np.ones((2, 3)), dims=("lat", "vertices"))

        result = validate_dataset(ds)

        assert any(
            issue.code == "invalid-bounds-vertex-count" and issue.variable == "lat_bnds"
            for issue in result.errors
        )


def test_validation_does_not_modify_dataset():
    ds = _create_rectilinear_dataset(with_bounds=False)
    original = ds.copy(deep=True)

    validate_dataset(ds)

    xrt.assert_identical(ds, original)


def test_validation_api_is_exported_from_top_level():
    assert xc.validate_dataset is validate_dataset
    assert xc.ValidationIssue is ValidationIssue
    assert xc.ValidationResult is ValidationResult
    assert xc.DatasetValidationError is DatasetValidationError
    assert not hasattr(xc, "ValidationOperation")
    assert not hasattr(xc, "ValidationSeverity")
