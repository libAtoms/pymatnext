"""Pydantic model for the ``[ns]`` section."""

from typing import Annotated, Optional

from pydantic import Field

from pymatnext.loop_exit.loop_exit_params import NSLoopExitParams
from pymatnext.params import PymatnextParams


class NSParams(PymatnextParams):
    """Parameters that control a nested-sampling calculation."""

    n_walkers: Annotated[int, Field(description="Number of NS walkers (live points)")]
    walk_length: Annotated[int, Field(description="Length of NS walk to produce a new, decorrelated config, in number of energy/force evaluations")]
    step_size_tune_walk_length: Annotated[Optional[int], Field(
        default=None,
        gt=0,
        description="NS walk length used to tune step size",
    )]
    configs_module: Annotated[str, Field(description="TODO: document the configuration implementation module.")]
    exit_conditions: Annotated[NSLoopExitParams, Field(
        default_factory=NSLoopExitParams,
        description="TODO: document optional loop-exit controls.",
    )]
    initial_config_file: Annotated[Optional[str], Field(
        default=None,
        description="TODO: document the initial configuration file.",
    )]
