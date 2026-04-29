from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union


AFC_CONFIG_VERSION = 1


class AFCConfigError(ValueError):
    """Raised when an AFC configuration file is malformed or incomplete."""


def _strip_inline_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for i, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if char == "#" and not in_single_quote and not in_double_quote:
            return line[:i].rstrip()

    return line.rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""

    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none", "~"}:
        return None

    if value[0:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError) as e:
                raise AFCConfigError(f"Could not parse inline YAML value: {value}") from e

    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value[1:-1]

    try:
        if any(char in lower for char in [".", "e"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Parses the small YAML subset used by the AFC example configs.

    This fallback intentionally supports only nested mappings plus JSON-style inline lists/dicts. If PyYAML is
    installed, `read_afc_config()` uses it instead.
    """
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if raw_line.strip() == "" or raw_line.lstrip().startswith("#"):
            continue

        line = _strip_inline_comment(raw_line)
        if line.strip() == "":
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise AFCConfigError(
                f"YAML fallback parser expects two-space indentation. Bad line {line_number}: {raw_line}"
            )

        stripped = line.strip()
        if ":" not in stripped:
            raise AFCConfigError(
                f"YAML fallback parser only supports key-value mappings. Bad line {line_number}: {raw_line}"
            )

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "":
            raise AFCConfigError(f"Empty YAML key on line {line_number}.")

        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _read_yaml(filepath: Path) -> Dict[str, Any]:
    text = filepath.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return {} if data is None else data
    except ImportError:
        return _parse_simple_yaml(text)


def merge_afc_configs(
    base: Optional[Mapping[str, Any]],
    *overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Recursively merges AFC config dictionaries.

    Later dictionaries override earlier dictionaries. Inputs are not mutated.
    """
    merged: Dict[str, Any] = deepcopy(dict(base or {}))

    for override in overrides:
        if override is None:
            continue
        for key, value in override.items():
            if (
                isinstance(value, Mapping)
                and isinstance(merged.get(key), Mapping)
            ):
                merged[key] = merge_afc_configs(merged[key], value)
            else:
                merged[key] = deepcopy(value)

    return merged


def get_afc_config_value(
    config: Mapping[str, Any],
    field_path: str,
    default: Any = None,
) -> Any:
    """
    Gets a nested config value using dot-separated keys, e.g. `"model.model_size"`.
    """
    current: Any = config
    for part in field_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def require_afc_config_fields(
    config: Mapping[str, Any],
    required_fields: Iterable[str],
) -> None:
    """
    Raises `AFCConfigError` if any dot-separated required config field is missing.
    """
    missing = [
        field for field in required_fields
        if get_afc_config_value(config, field, default=None) is None
    ]
    if missing:
        raise AFCConfigError(f"AFC config is missing required fields: {missing}")


def read_afc_config(
    filepath: Union[str, Path],
    *,
    defaults: Optional[Mapping[str, Any]] = None,
    required_fields: Iterable[str] = (),
) -> Dict[str, Any]:
    """
    Reads an AFC config file from JSON or YAML.

    YAML support uses PyYAML when available. Without PyYAML, a small built-in parser handles the example config subset:
    nested mappings with two-space indentation and JSON-style inline lists/dicts.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        loaded = _read_yaml(filepath)
    else:
        raise ValueError(
            f'Unsupported AFC config extension "{filepath.suffix}". Use `.json`, `.yaml`, or `.yml`.'
        )

    if not isinstance(loaded, Mapping):
        raise AFCConfigError("Top-level AFC config must be a mapping/object.")

    config = merge_afc_configs(defaults, loaded)
    require_afc_config_fields(config, required_fields)
    return config


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return json.dumps(str(value))


def _to_simple_yaml(data: Mapping[str, Any], indent: int = 0) -> str:
    lines = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f"{prefix}{key}:")
            lines.append(_to_simple_yaml(value, indent=indent + 2))
        else:
            lines.append(f"{prefix}{key}: {_format_yaml_scalar(value)}")
    return "\n".join(lines)


def write_afc_config(
    config: Mapping[str, Any],
    filepath: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    """
    Writes an AFC config to JSON or a conservative YAML subset.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = filepath.suffix.lower()

    if suffix == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=indent)
    elif suffix in {".yaml", ".yml"}:
        filepath.write_text(_to_simple_yaml(config) + "\n", encoding="utf-8")
    else:
        raise ValueError(
            f'Unsupported AFC config extension "{filepath.suffix}". Use `.json`, `.yaml`, or `.yml`.'
        )

    return filepath
