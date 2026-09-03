"""Pydantic model for the ``[ns.exit_conditions]`` section."""

from typing import Annotated, Any, Dict, Optional

from pydantic import Field

from pymatnext.params import PymatnextParams


class NSLoopExitParams(PymatnextParams):
    """Parameters for an optional nested-sampling loop exit evaluator."""

    module: Annotated[Optional[str], Field(default=None, description="TODO: document the loop-exit evaluator module.")]
    module_kwargs: Annotated[Dict[str, Any], Field(
        default_factory=dict,
        description="TODO: document keyword arguments for the loop-exit evaluator.",
    )]
