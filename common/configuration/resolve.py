import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Generic, TypeVar

from common.configuration.provider import ConfigProvider

from .interface import ConfigLike
from .provider import get_config_provider

T = TypeVar("T")


@dataclass
class ConfigValue(Generic[T]):
    """A value that can be resolved from a configuration store."""

    key: str | type[T]
    default: T | None = None
    description: str | None = None
    after: Callable[[T], T] | None = None
    mandatory: bool = False

    @property
    def value(self) -> T:
        return self.resolve()

    def resolve(self, context: str | None = None, **kwargs: Any) -> T:
        provider: ConfigProvider = get_config_provider()
        config: ConfigLike = provider.get_config(context)

        val: T | None
        if inspect.isclass(self.key):
            val = self.key(**kwargs)
        else:
            path: list[str] = [p.strip() for p in str(self.key).split(",")]
            val = config.get(*path, default=self.default)

        if val is None:
            if self.mandatory:
                raise ValueError(f"ConfigValue '{self.key}' is mandatory but missing from config")
            return val  # type: ignore

        if self.after is not None:
            val = self.after(val)

        return val

    @staticmethod
    def create_field(key: str, default: Any = None, description: str | None = None) -> Any:
        """Create a field for a dataclass that will be resolved at creation time."""
        return field(
            default_factory=lambda: ConfigValue(key=key, default=default, description=description).resolve()
        )  # pylint: disable=invalid-field-call


def resolve_arguments(fn_or_cls, args, kwargs):
    """
    Replace any ConfigValue arguments (positional or keyword) with their resolved values.
    If a parameter has a default that is a ConfigValue and the caller didn't supply it,
    resolve that default.
    """

    sig = inspect.signature(fn_or_cls)
    ba = sig.bind_partial(*args, **kwargs)

    # Resolve provided arguments
    for name, value in list(ba.arguments.items()):
        if isinstance(value, ConfigValue):
            ba.arguments[name] = value.resolve()

    # Resolve default ConfigValue for missing args
    for name, param in sig.parameters.items():
        if name not in ba.arguments and isinstance(param.default, ConfigValue):
            ba.arguments[name] = param.default.resolve()

    return ba.args, ba.kwargs


def inject_config(fn_or_cls):
    @functools.wraps(fn_or_cls)
    def decorated(*args, **kwargs):
        args, kwargs = resolve_arguments(fn_or_cls, args, kwargs)
        return fn_or_cls(*args, **kwargs)

    return decorated


class Configurable:
    """A base class for dataclasses that can have ConfigValue fields."""

    def resolve(self):
        """Resolve all ConfigValue fields in the dataclass."""
        if not is_dataclass(self):
            return
        for attrib in fields(self):
            if isinstance(getattr(self, attrib.name), ConfigValue):
                setattr(self, attrib.name, getattr(self, attrib.name).resolve())

    # def __post_init__(self):
    #     self.resolve()
