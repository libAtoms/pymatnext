"""Pydantic model for the ``[ns]`` section."""

from pydantic import Field

from pymatnext.loop_exit.loop_exit_params import NSLoopExitParams
from pymatnext.params import PymatnextParams


class NSParams(PymatnextParams):
    """Parameters that control a nested-sampling calculation."""

    n_walkers: int
    walk_length: int
    step_size_tune_walk_length: int = 0
    configs_module: str
    exit_conditions: NSLoopExitParams = Field(default_factory=NSLoopExitParams)
    initial_config_file: str = "_NONE_"
