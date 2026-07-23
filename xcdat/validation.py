"""Dataset validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import cf_xarray  # noqa: F401
import xarray as xr

from xcdat.axis import CFAxisKey, CF_ATTR_MAP, VAR_NAME_MAP

_Severity = Literal["error", "warning"]
_AFFECTED_OPERATIONS = (
    "spatial_average",
    "temporal",
    "horizontal_regrid",
    "vertical_regrid",
)
_AXIS_OPERATIONS: dict[CFAxisKey, tuple[str, ...]] = {
    "X": ("spatial_average", "horizontal_regrid"),
    "Y": ("spatial_average", "horizontal_regrid"),
    "T": ("temporal",),
    "Z": ("vertical_regrid",),
}


@dataclass(frozen=True)
class ValidationIssue:
    """A dataset compatibility issue.

    Parameters
    ----------
    code : str
        Stable identifier for the detected problem.
    severity : {"error", "warning"}
        Problem severity.
    variable : str
        Affected variable or coordinate.
    problem : str
        Description of the detected problem.
    operations : tuple[str, ...]
        xCDAT operations that may be affected.
    suggestion : str
        Suggested user action.
    """

    code: str
    severity: _Severity
    variable: str
    problem: str
    operations: tuple[str, ...]
    suggestion: str


@dataclass(frozen=True)
class ValidationResult:
    """Results returned by :func:`validate_dataset`.

    Parameters
    ----------
    issues : tuple[ValidationIssue, ...]
        Detected validation issues.
    """

    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return issues classified as errors."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return issues classified as warnings."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Return whether validation found no errors."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise an exception containing all validation errors.

        Raises
        ------
        DatasetValidationError
            If one or more validation errors were found.
        """
        if self.errors:
            raise DatasetValidationError(self.errors)


class DatasetValidationError(ValueError):
    """Error raised for invalid xCDAT dataset metadata."""

    def __init__(self, issues: tuple[ValidationIssue, ...]):
        self.issues = issues
        details = "\n".join(
            f"- [{issue.code}] {issue.variable}: {issue.problem}" for issue in issues
        )
        super().__init__(f"Dataset validation failed:\n{details}")


class _IssueCollector:
    """Collect and merge validation issues."""

    def __init__(self) -> None:
        self._issues: dict[tuple[str, str, str], ValidationIssue] = {}

    def add(
        self,
        code: str,
        severity: _Severity,
        variable: str,
        problem: str,
        operations: Iterable[str],
        suggestion: str,
    ) -> None:
        key = (code, variable, problem)
        current = self._issues.get(key)
        operation_set = set(operations)

        if current is not None:
            operation_set.update(current.operations)

        ordered_operations = tuple(
            operation
            for operation in _AFFECTED_OPERATIONS
            if operation in operation_set
        )
        self._issues[key] = ValidationIssue(
            code=code,
            severity=severity,
            variable=variable,
            problem=problem,
            operations=ordered_operations,
            suggestion=suggestion,
        )

    def result(self) -> ValidationResult:
        issues = tuple(
            sorted(
                self._issues.values(),
                key=lambda issue: (issue.code, issue.variable, issue.problem),
            )
        )
        return ValidationResult(issues)


def validate_dataset(ds: xr.Dataset) -> ValidationResult:
    """Validate dataset coordinate and bounds metadata.

    Existing malformed or contradictory metadata is reported as an error.
    Missing metadata that xCDAT may be able to generate or infer is reported as
    a warning. Validation does not modify the dataset or load array data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.

    Returns
    -------
    ValidationResult
        Structured validation diagnostics.

    Raises
    ------
    TypeError
        If ``ds`` is not an ``xarray.Dataset``.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("The 'ds' argument must be an xarray.Dataset.")

    collector = _IssueCollector()
    axis_coords = _get_axis_coords(ds)
    cf_axis_coords = _get_cf_axis_coords(ds)

    _validate_axis_metadata(ds, axis_coords, cf_axis_coords, collector)
    _validate_bounds(ds, axis_coords, collector)

    return collector.result()


def _get_cf_axis_coords(ds: xr.Dataset) -> dict[CFAxisKey, set[str]]:
    result: dict[CFAxisKey, set[str]] = {axis: set() for axis in CF_ATTR_MAP}

    for axis, attrs in CF_ATTR_MAP.items():
        result[axis].update(str(key) for key in ds.cf.axes.get(attrs["axis"], []))
        result[axis].update(
            str(key) for key in ds.cf.coordinates.get(attrs["coordinate"], [])
        )

    return result


def _get_axis_coords(ds: xr.Dataset) -> dict[CFAxisKey, set[str]]:
    result = _get_cf_axis_coords(ds)

    for axis, names in VAR_NAME_MAP.items():
        result[axis].update(name for name in names if name in ds.coords)

    return result


def _validate_axis_metadata(
    ds: xr.Dataset,
    axis_coords: dict[CFAxisKey, set[str]],
    cf_axis_coords: dict[CFAxisKey, set[str]],
    collector: _IssueCollector,
) -> None:
    conflicting_coords = _validate_axis_conflicts(axis_coords, collector)

    for axis, coords in axis_coords.items():
        for coord in coords.difference(cf_axis_coords[axis]):
            if coord in conflicting_coords:
                continue
            collector.add(
                "missing-cf-axis-metadata",
                "warning",
                coord,
                f"Coordinate is inferred by name but is not mapped to the '{axis}' axis "
                "by cf_xarray.",
                _AXIS_OPERATIONS[axis],
                "Set consistent CF 'axis', 'standard_name', and 'units' attributes.",
            )

    for name, data_var in ds.data_vars.items():
        for axis, coords in axis_coords.items():
            dim_coords = [
                coord
                for coord in coords
                if coord in ds.indexes
                and ds[coord].ndim == 1
                and set(ds[coord].dims).issubset(data_var.dims)
            ]
            if len(dim_coords) > 1:
                collector.add(
                    "ambiguous-axis",
                    "error",
                    str(name),
                    "Data variable has multiple dimension coordinates "
                    f"{sorted(dim_coords)} mapped to the '{axis}' axis.",
                    _AXIS_OPERATIONS[axis],
                    "Correct conflicting coordinate metadata or select one coordinate "
                    "system before using xCDAT operations.",
                )


def _validate_axis_conflicts(
    axis_coords: dict[CFAxisKey, set[str]], collector: _IssueCollector
) -> set[str]:
    coord_axes: dict[str, list[CFAxisKey]] = {}
    for axis, coords in axis_coords.items():
        for coord in coords:
            coord_axes.setdefault(coord, []).append(axis)

    conflicting_coords: set[str] = set()
    for coord, axes in coord_axes.items():
        if len(axes) > 1:
            conflicting_coords.add(coord)
            collector.add(
                "conflicting-axis-metadata",
                "error",
                coord,
                f"Coordinate is mapped or inferred to multiple axes: {sorted(axes)}.",
                (operation for axis in axes for operation in _AXIS_OPERATIONS[axis]),
                "Correct the coordinate name or its conflicting CF axis attributes.",
            )

    return conflicting_coords


def _validate_bounds(
    ds: xr.Dataset,
    axis_coords: dict[CFAxisKey, set[str]],
    collector: _IssueCollector,
) -> None:
    coord_axes = {
        coord: axis for axis, coords in axis_coords.items() for coord in coords
    }
    for coord_name, axis in sorted(coord_axes.items()):
        coord = ds[coord_name]
        bounds_name = coord.attrs.get("bounds")
        operations = _AXIS_OPERATIONS[axis]

        if not isinstance(bounds_name, str) or not bounds_name.strip():
            collector.add(
                "missing-bounds",
                "warning",
                coord_name,
                "Coordinate does not reference a bounds variable.",
                operations,
                "Add bounds explicitly or use 'ds.bounds.add_bounds()' or "
                "'ds.bounds.add_missing_bounds()'.",
            )
            continue

        bounds_name = bounds_name.strip()
        if bounds_name not in ds.variables:
            collector.add(
                "missing-bounds-variable",
                "error",
                coord_name,
                f"Bounds attribute references missing variable '{bounds_name}'.",
                operations,
                "Add the referenced bounds variable or correct the 'bounds' attribute.",
            )
            continue

        bounds = ds[bounds_name]
        missing_dims = set(coord.dims).difference(bounds.dims)
        if missing_dims:
            collector.add(
                "bounds-missing-coordinate-dimensions",
                "error",
                bounds_name,
                "Bounds variable is missing coordinate dimensions "
                f"{sorted(str(dim) for dim in missing_dims)}.",
                operations,
                "Make bounds include every dimension of the related coordinate.",
            )
            continue

        mismatched_dims = [
            dim for dim in coord.dims if coord.sizes[dim] != bounds.sizes[dim]
        ]
        if mismatched_dims:
            collector.add(
                "bounds-dimension-size-mismatch",
                "error",
                bounds_name,
                "Bounds sizes do not match coordinate sizes for dimensions "
                f"{sorted(str(dim) for dim in mismatched_dims)}.",
                operations,
                "Make coordinate and bounds dimension sizes identical.",
            )

        extra_dims = [dim for dim in bounds.dims if dim not in coord.dims]
        if len(extra_dims) != 1:
            collector.add(
                "malformed-bounds-dimensions",
                "error",
                bounds_name,
                "Bounds variable must have exactly one dimension in addition to its "
                f"coordinate dimensions; found {extra_dims}.",
                operations,
                "Remove unrelated dimensions and retain one bounds vertex dimension.",
            )
            continue

        expected_vertices = 2 if coord.ndim == 1 else 4 if coord.ndim == 2 else None
        if (
            expected_vertices is not None
            and bounds.sizes[extra_dims[0]] != expected_vertices
        ):
            collector.add(
                "invalid-bounds-vertex-count",
                "error",
                bounds_name,
                f"Bounds vertex dimension has size {bounds.sizes[extra_dims[0]]}; "
                f"expected {expected_vertices} for a {coord.ndim}-D coordinate.",
                operations,
                "Provide the CF-compatible number of vertices for the coordinate.",
            )
