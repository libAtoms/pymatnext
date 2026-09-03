"""Pydantic model for the ``[ns]`` section."""

from typing import Optional

from pydantic import Field

from pymatnext.loop_exit.loop_exit_params import NSLoopExitParams
from pymatnext.params import PymatnextParams


class NSParams(PymatnextParams):
    """Parameters that control a nested-sampling calculation."""

    n_walkers: int = Field(..., description="TODO: document the number of nested-sampling walkers.")
    walk_length: int = Field(..., description="TODO: document the number of walk steps per iteration.")
    step_size_tune_walk_length: int = Field(
        0,
        description="TODO: document the walk length used for step-size tuning.",
    )
    configs_module: str = Field(..., description="TODO: document the configuration implementation module.")
    exit_conditions: NSLoopExitParams = Field(
        default_factory=NSLoopExitParams,
        description="TODO: document optional loop-exit controls.",
    )
    initial_config_file: Optional[str] = Field(
        None,
        description="TODO: document the initial configuration file.",
    )
