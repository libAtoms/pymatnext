"""Top-level Pydantic model for nested-sampling parameters."""

from typing import Any, List

import toml
from pydantic import Field

from pymatnext.config_utils import apply_override, deep_update, load_packaged_toml, parse_override
from pymatnext.ns_configs.ase_atoms.ase_atoms_params import ASEAtomsParams
from pymatnext.ns_params import NSParams
from pymatnext.params import PymatnextParams


DEFAULTS_RESOURCE = "sample_defaults.toml"


class StepSizeTuneParams(PymatnextParams):
    interval: int = 1000
    n_configs: int = 1
    min_accept_rate: float = 0.25
    max_accept_rate: float = 0.5
    adjust_factor: float = 1.25


class WalkTrajectoryInfoParams(PymatnextParams):
    iter_min: int = -1
    iter_max: int = -1
    interval: int = 0
    avg_times: List[Any] = Field(default_factory=list)


class GlobalParams(PymatnextParams):
    output_filename_prefix: str = "NS"
    output_filename_prefix_extra: str = ""
    random_seed: int = -1
    max_iter: int = -1
    stdout_report_interval_s: int = 60
    sample_interval: int = 1
    traj_interval: int = 100
    snapshot_interval: int = 10000
    snapshot_save_old: int = 2
    step_size_tune: StepSizeTuneParams = Field(default_factory=StepSizeTuneParams)
    walk_traj_info: WalkTrajectoryInfoParams = Field(default_factory=WalkTrajectoryInfoParams)
    clone_history: bool = False
    override_initial_max_val: bool = False
    initial_max_val: float = 0.0


class SampleParams(PymatnextParams):
    """All parameters consumed by :func:`pymatnext.cli.sample.sample`."""

    global_: GlobalParams = Field(default_factory=GlobalParams, alias="global")
    ns: NSParams
    configs: ASEAtomsParams

    @classmethod
    def default_data(cls):
        """Return the model-default layer in TOML-compatible key form."""

        return {"global": GlobalParams().model_dump()}


def load_sample_params(input_file, overrides):
    """Load parameter layers and return the validated top-level model.

    The precedence is Pydantic model defaults, package TOML, runtime TOML,
    then command-line overrides.
    """

    data = SampleParams.default_data()
    deep_update(data, load_packaged_toml("pymatnext", DEFAULTS_RESOURCE))

    with open(input_file) as fin:
        deep_update(data, toml.load(fin))

    for spec in overrides:
        path, value = parse_override(spec)
        apply_override(data, path, value)

    return SampleParams.model_validate(data)
