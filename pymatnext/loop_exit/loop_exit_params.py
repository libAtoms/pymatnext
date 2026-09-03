"""Pydantic model for the ``[ns.exit_conditions]`` section."""

from typing import Any, Dict

from pydantic import Field

from pymatnext.params import PymatnextParams


class NSLoopExitParams(PymatnextParams):
    """Parameters for an optional nested-sampling loop exit evaluator."""

    module: str = "_NONE_"
    module_kwargs: Dict[str, Any] = Field(default_factory=dict)
