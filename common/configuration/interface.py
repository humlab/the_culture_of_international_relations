from typing import Any, Protocol, runtime_checkable

# pylint: disable=unused-argument


@runtime_checkable
class ConfigLike(Protocol):
    def get(self, *keys: str, default: Any | type[Any] = None, mandatory: bool = False) -> Any: ...
    def exists(self, *keys: str) -> bool: ...
    def update(self, data: tuple[str, Any] | dict[str, Any] | list[tuple[str, Any]]) -> None: ...


@runtime_checkable
class ConfigFactoryLike(Protocol):
    @staticmethod
    def load(
        *,
        source: str | dict | ConfigLike | None = None,
        context: str | None = None,
        env_filename: str | None = None,
        env_prefix: str | None = None,
    ) -> ConfigLike: ...
    @staticmethod
    def is_config_path(source: Any) -> bool: ...


# def is_config_like(obj: Any) -> TypeGuard[ConfigLike]:
#     return all(callable(getattr(obj, name, None)) for name in ("get", "exists", "update", "add"))


# def is_config_factory_like(cls: Any) -> bool:
#     return all(callable(getattr(cls, name, None)) for name in ("load", "is_config_path"))
