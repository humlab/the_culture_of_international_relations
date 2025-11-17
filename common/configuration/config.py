from __future__ import annotations

import io
from inspect import isclass
from os.path import join, normpath
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from common.configuration.utility import dget, dotexists, dotset, env2dict, replace_env_vars

# pylint: disable=too-many-arguments


def yaml_str_join(loader: yaml.Loader, node: yaml.SequenceNode) -> str:
    return "".join([str(i) for i in loader.construct_sequence(node)])


def yaml_path_join(loader: yaml.Loader, node: yaml.SequenceNode) -> str:
    return join(*[str(i) for i in loader.construct_sequence(node)])


def nj(*paths: str) -> str:
    return normpath(join(*paths)) if None not in paths else ""


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
    def let_unknown_through(self, node):  # pylint: disable=unused-argument
        """Ignore unknown tags silently"""
        if isinstance(node, yaml.ScalarNode):
            return self.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return self.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return self.construct_mapping(node)
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.let_unknown_through)  # type: ignore
SafeLoaderIgnoreUnknown.add_constructor("!join", yaml_str_join)
SafeLoaderIgnoreUnknown.add_constructor("!jj", yaml_path_join)
SafeLoaderIgnoreUnknown.add_constructor("!path_join", yaml_path_join)


class Config:
    """Container for configuration elements."""

    def __init__(
        self,
        *,
        data: dict | None = None,
        context: str = "default",
        filename: str | None = None,
    ):
        self.data: dict | None = data
        self.context: str = context
        self.filename: str | None = filename

    def get(self, *keys: str, default: Any | type[Any] = None, mandatory: bool = False) -> Any:
        if self.data is None:
            raise ValueError("Configuration not initialized")

        if mandatory and not self.exists(*keys):
            raise ValueError(f"Missing mandatory key: {'/'.join(keys)}")

        value: Any = dget(self.data, *keys)

        if value is not None:
            return value

        if callable(default) and not isinstance(default, type):
            return default()

        # Allow instance of class to be returned by calling default (parameterless) constructor
        return default() if isclass(default) else default

    def update(self, data: tuple[str, Any] | dict[str, Any] | list[tuple[str, Any]]) -> None:
        if self.data is None:
            self.data = {}
        items = [data] if isinstance(data, tuple) else data.items() if isinstance(data, dict) else data
        for key, value in items:
            dotset(self.data, key, value)

    def exists(self, *keys: str) -> bool:
        return False if self.data is None else dotexists(self.data, *keys)  # type: ignore


class ConfigFactory:
    """Factory for creating Config instances."""

    def load(
        self,
        *,
        source: str | dict | Config | None = None,
        context: str | None = None,
        env_filename: str | None = None,
        env_prefix: str | None = None,
    ) -> Config:

        load_dotenv(dotenv_path=env_filename)

        if isinstance(source, Config):
            return source

        if source is None:
            source = {}

        data: dict = (
            (
                yaml.load(
                    Path(source).read_text(encoding="utf-8"),
                    Loader=SafeLoaderIgnoreUnknown,
                )
                if self.is_config_path(source, raise_if_missing=True)
                else yaml.load(io.StringIO(source), Loader=SafeLoaderIgnoreUnknown)
            )
            if isinstance(source, str)
            else source
        )

        # Handle empty YAML files (loads as None)
        data = data or {}

        if not isinstance(data, dict):
            raise TypeError(f"expected dict, found '{type(data)}'")

        # Resolve sub-configurations by loading referenced files recursively

        data = self._resolve_sub_configs(
            data,
            context=context,
            env_filename=env_filename,
            env_prefix=env_prefix,
            source_path=source if isinstance(source, str) else None,
        )

        # Update data based on environment variables with a name that starts with `env_prefix`
        assert isinstance(env_prefix, str)
        data = env2dict(env_prefix, data)

        # Do a recursive replace of values with pattern "${ENV_NAME}" with value of environment
        data = replace_env_vars(data)

        return Config(
            data=data,
            context=context or "default",
            filename=source if self.is_config_path(source) else None,  # type: ignore
        )

    @staticmethod
    def is_config_path(source: Any, raise_if_missing: bool = True) -> bool:
        """Test if the source is a valid path to a configuration file."""
        if not isinstance(source, str):
            return False
        if not source.endswith(".yaml") and not source.endswith(".yml"):
            return False
        if raise_if_missing and not Path(source).exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")
        return True

    def _resolve_sub_configs(
        self,
        data: dict,
        *,
        context: str | None = None,
        env_filename: str | None = None,
        env_prefix: str | None = None,
        source_path: str | None = None,
    ) -> dict:
        """Recursively resolve sub-configurations referenced in the main configuration.

        A sub-config is referenced using the @include: prefix, e.g. "@include:path/to/subconfig.yaml"
        Relative paths are resolved relative to the main configuration file.
        Sub-configs can themselves reference further sub-configs.

        Example:
            database: "@include:config/database.yml"
            api: "@include:config/api.yml"
        """

        def _resolve(value: Any, base_path: Path | None) -> Any:
            if isinstance(value, dict):
                return {k: _resolve(v, base_path) for k, v in value.items()}
            if isinstance(value, list):
                return [_resolve(v, base_path) for v in value]
            if isinstance(value, str) and value.startswith("@include:"):
                sub_config_path = value[9:]  # Remove "@include:" prefix
                if not Path(sub_config_path).is_absolute() and base_path is not None:
                    sub_config_path = str(base_path.parent / sub_config_path)
                loaded_data = (
                    ConfigFactory()
                    .load(source=sub_config_path, context=context, env_filename=env_filename, env_prefix=env_prefix)
                    .data
                )
                return _resolve(loaded_data, Path(sub_config_path))
            return value

        # Determine base path from source_path if available
        initial_base_path = (
            Path(source_path) if source_path and self.is_config_path(source_path, raise_if_missing=False) else None
        )
        return _resolve(data, initial_base_path)
