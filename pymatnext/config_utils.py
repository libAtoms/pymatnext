"""Utilities for merging and documenting TOML configuration sources."""

from copy import deepcopy
from enum import Enum
from importlib.resources import files
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import toml
from pydantic import BaseModel


NONE_SENTINEL = "_NONE_"


def load_packaged_toml(package: str, resource: str) -> dict[str, Any]:
    """Load a TOML resource shipped with *package*."""

    with files(package).joinpath(resource).open("r") as fin:
        return decode_toml_none(toml.load(fin))


def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base* and return *base*."""

    for key, value in overlay.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def parse_override(spec: str) -> tuple[str, Any]:
    """Parse a ``dotted.path=TOML_LITERAL`` command-line override."""

    if "=" not in spec:
        raise ValueError(f"Override must be KEY=VALUE (got {spec!r})")
    path, raw_value = spec.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError("Override path must not be empty")
    try:
        value = toml.loads("_ = " + raw_value.strip())["_"]
    except toml.TomlDecodeError as exc:
        raise ValueError(f"Could not parse override value as TOML: {raw_value!r}") from exc
    return path, decode_toml_none(value)


def apply_override(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value in a nested mapping, creating intermediate mappings as needed."""

    cur = data
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def decode_toml_none(value: Any) -> Any:
    """Convert ``_NONE_`` TOML values recursively to Python ``None``."""

    if value == NONE_SENTINEL:
        return None
    if isinstance(value, dict):
        return {key: decode_toml_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_toml_none(item) for item in value]
    return value


def _toml_value(value: Any) -> str:
    """Render a single Python value using TOML literal syntax."""

    if value is None:
        return f'"{NONE_SENTINEL}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, dict):
        items = ", ".join(f'"{key}" = {_toml_value(item)}' for key, item in value.items())
        return "{ " + items + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _strip_annotated(annotation: Any) -> Any:
    """Unwrap ``Annotated[T, ...]`` down to ``T``."""

    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _field_choices(annotation: Any):
    """Return allowed literal or enum values for an annotation, if specified."""

    annotation = _strip_annotated(annotation)
    members = [annotation]
    if get_origin(annotation) is Union:
        members = get_args(annotation)

    for member in members:
        member = _strip_annotated(member)
        if get_origin(member) is Literal:
            return list(get_args(member))
        if isinstance(member, type) and issubclass(member, Enum):
            return [item.value for item in member]
    return None


def _format_annotation(annotation: Any) -> str:
    """Return a human-readable type description."""

    annotation = _strip_annotated(annotation)
    if get_origin(annotation) is Union:
        members = [member for member in get_args(annotation) if member is not type(None)]
        return " | ".join(_format_annotation(member) for member in members)
    if get_origin(annotation) is Literal:
        return "Literal[" + ", ".join(repr(item) for item in get_args(annotation)) + "]"
    return getattr(annotation, "__name__", str(annotation))


def _model_type(annotation: Any):
    """Return the Pydantic model represented by an annotation, if any."""

    annotation = _strip_annotated(annotation)
    if get_origin(annotation) is Union:
        for member in get_args(annotation):
            model = _model_type(member)
            if model is not None:
                return model
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _field_default(field):
    """Return a field's default, evaluating its default factory when present."""

    if field.default_factory is not None:
        return field.default_factory()
    return field.default


def _field_help(name: str, field) -> list:
    """Render comments and an assignment placeholder for one scalar field."""

    lines = []
    if field.description:
        lines.append(f"## {field.description}")
    lines.append(f"## type: {_format_annotation(field.annotation)}")

    choices = _field_choices(field.annotation)
    if choices is not None:
        lines.append("## choices: " + ", ".join(_toml_value(choice) for choice in choices))

    if field.is_required():
        lines.append(f"# {name} =   # REQUIRED")
    else:
        default = _field_default(field)
        lines.append(f"{name} = {_toml_value(default)}")
    lines.append("")
    return lines


def _toml_help(model: type, table_path: tuple[str, ...] = ()) -> list:
    """Recursively render a Pydantic model as an annotated TOML template."""

    scalar_fields = []
    model_fields = []
    for name, field in model.model_fields.items():
        field_name = field.alias or name
        model_type = _model_type(field.annotation)
        if model_type is None:
            scalar_fields.append((field_name, field))
        else:
            model_fields.append((field_name, field, model_type))

    lines = []
    for name, field in scalar_fields:
        lines.extend(_field_help(name, field))

    for name, field, nested_model in model_fields:
        if lines and lines[-1] != "":
            lines.append("")
        if field.description:
            lines.append(f"## {field.description}")
        lines.append(f"## type: {_format_annotation(field.annotation)}")
        lines.append("[" + ".".join(table_path + (name,)) + "]")
        lines.extend(_toml_help(nested_model, table_path + (name,)))

    return lines


def format_toml_help(model: type) -> str:
    """Return an annotated TOML template generated from Pydantic field metadata."""

    return "\n".join(_toml_help(model)).rstrip()


def format_defaults(model: type, package_defaults: dict[str, Any]) -> str:
    """Return package defaults and a documented TOML template for *model*."""

    lines = ["config file defaults:"]
    for key, value in package_defaults.items():
        lines.append(f"    {key} = {_toml_value(value)}")
    lines.extend([
        "",
        "config file fields:",
        f'## The TOML string "{NONE_SENTINEL}" represents Python None.',
        format_toml_help(model),
    ])
    return "\n".join(lines)
