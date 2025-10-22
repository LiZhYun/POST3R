"""Utility functions for POST3R"""
import itertools
from typing import Any, Dict, Iterable, Optional


def config_as_kwargs(
    config, to_filter: Optional[Iterable[str]] = None, defaults: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Build kwargs for constructor from config dictionary."""
    always_filter = ("name",)
    if to_filter:
        to_filter = tuple(itertools.chain(always_filter, to_filter))
    else:
        to_filter = always_filter
    if defaults:
        # Defaults come first such that they can be overwritten by config
        to_iter = itertools.chain(defaults.items(), config.items())
    else:
        to_iter = config.items()
    return {k: v for k, v in to_iter if k not in to_filter}
