"""Pydantic model for the ``[ns]`` section."""

from typing import Annotated, Optional

from pydantic import Field, PositiveInt

from pymatnext.loop_exit.loop_exit_params import NSLoopExitParams
from pymatnext.params import PymatnextParams


class NSParams(PymatnextParams):
    """Parameters that control a nested-sampling calculation."""

    n_walkers: Annotated[int, Field(description="Number of NS walkers (live points)")]
    walk_length: Annotated[int, Field(description="Length of NS walk to produce a new, decorrelated config, in number of energy/force evaluations")]
    step_size_tune_walk_length: Annotated[PositiveInt | None, Field(
        default=None,
        description="NS walk length used to tune step size",
    )]
    configs_module: Annotated[str, Field(description="module that defines configurations")]
    exit_conditions: Annotated[NSLoopExitParams, Field(
        default_factory=NSLoopExitParams,
        description="parameters for optional exit conditions",
    )]
    initial_config_file: Annotated[str | None, Field(
        default=None,
        description="file with initial configurations, if not generated randomly",
    )]
    initial_max_val: Annotated[float | None, Field(default=None, description="value of overriding NS energy initial maxmimum")]
