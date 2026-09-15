"""Pydantic model for the ``[ns.exit_conditions]`` section."""

from __future__ import annotations
from typing import Annotated, Any

from pydantic import Field

from pymatnext.params import PymatnextParams


class NSLoopExitParams(PymatnextParams):
    """Parameters for an optional nested-sampling loop exit evaluator."""

    module: Annotated[str | None, Field(default=None, description="module that defines an `ExitLoop` class that checks for an exit condition, with `__call__` method that takes current iteration and max value and returns a bool")]
    module_kwargs: Annotated[dict[str, Any], Field(
        default_factory=dict,
        description="arbitrary kwargs for `ExitLoope constructor in addition to the `NS` object itself",
    )]
