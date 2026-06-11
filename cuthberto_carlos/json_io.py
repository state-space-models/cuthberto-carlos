"""JSON helpers for saving and loading JAX array trees."""

import importlib
import json
from typing import Any, cast

from jax import numpy as jnp


def _type_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _type_from_path(path: str) -> type:
    module_name, qualname = path.split(":", maxsplit=1)
    obj: Any = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return cast(type, obj)


def _to_jsonable(tree):
    if isinstance(tree, tuple) and hasattr(tree, "_asdict"):
        return {
            "__namedtuple__": _type_path(tree.__class__),
            **{key: _to_jsonable(value) for key, value in tree._asdict().items()},
        }
    if isinstance(tree, dict):
        return {key: _to_jsonable(value) for key, value in tree.items()}
    return jnp.asarray(tree).tolist()


def _from_jsonable(tree):
    if isinstance(tree, dict) and "__namedtuple__" in tree:
        cls = _type_from_path(tree["__namedtuple__"])
        return cls(
            **{
                key: _from_jsonable(value)
                for key, value in tree.items()
                if key != "__namedtuple__"
            }
        )
    if isinstance(tree, dict):
        return {key: _from_jsonable(value) for key, value in tree.items()}
    return jnp.asarray(tree)


def save_arraytree(tree, path: str) -> None:
    """Save a JAX array tree containing dicts and NamedTuples to JSON."""
    with open(path, "w") as f:
        json.dump(_to_jsonable(tree), f, indent=2)


def load_arraytree(path: str):
    """Load a JAX array tree saved by ``save_arraytree`` from JSON."""
    with open(path, "r") as f:
        return _from_jsonable(json.load(f))
