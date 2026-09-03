"""Top-level Pydantic model for nested-sampling parameters."""

from typing import Any, List

import toml
from pydantic import Field

from pymatnext.config_utils import (
    apply_override,
    decode_toml_none,
    deep_update,
    format_defaults,
    load_packaged_toml,
    parse_override,
)
from pymatnext.ns_configs.ase_atoms.ase_atoms_params import ASEAtomsParams
from pymatnext.ns_params import NSParams
from pymatnext.params import PymatnextParams


DEFAULTS_RESOURCE = "sample_defaults.toml"


class StepSizeTuneParams(PymatnextParams):
    interval: int = Field(1000, description="TODO: document the step-size tuning interval.")
    n_configs: int = Field(1, description="TODO: document the number of configurations for tuning.")
    min_accept_rate: float = Field(0.25, description="TODO: document the minimum accepted move rate.")
    max_accept_rate: float = Field(0.5, description="TODO: document the maximum accepted move rate.")
    adjust_factor: float = Field(1.25, description="TODO: document the tuning adjustment factor.")


class WalkTrajectoryInfoParams(PymatnextParams):
    iter_min: int = Field(-1, description="TODO: document the first iteration that writes walk trajectories.")
    iter_max: int = Field(-1, description="TODO: document the final iteration that writes walk trajectories.")
    interval: int = Field(0, description="TODO: document the walk-trajectory output interval.")
    avg_times: List[Any] = Field(default_factory=list, description="TODO: document walk-trajectory averaging times.")


class GeneralParams(PymatnextParams):
    output_filename_prefix: str = Field("NS", description="TODO: document the output filename prefix.")
    output_filename_prefix_extra: str = Field("", description="TODO: document text appended to output filenames.")
    random_seed: int = Field(-1, description="TODO: document the random-number seed.")
    max_iter: int = Field(-1, description="TODO: document the maximum nested-sampling iterations.")
    stdout_report_interval_s: int = Field(60, description="TODO: document the stdout reporting interval in seconds.")
    sample_interval: int = Field(1, description="TODO: document the NS-sample output interval.")
    traj_interval: int = Field(100, description="TODO: document the trajectory output interval.")
    snapshot_interval: int = Field(10000, description="TODO: document the snapshot output interval.")
    snapshot_save_old: int = Field(2, description="TODO: document how many old snapshots to retain.")
    step_size_tune: StepSizeTuneParams = Field(
        default_factory=StepSizeTuneParams,
        description="TODO: document step-size tuning controls.",
    )
    walk_traj_info: WalkTrajectoryInfoParams = Field(
        default_factory=WalkTrajectoryInfoParams,
        description="TODO: document walk-trajectory output controls.",
    )
    clone_history: bool = Field(False, description="TODO: document clone-history output.")
    override_initial_max_val: bool = Field(False, description="TODO: document initial maximum-value override.")
    initial_max_val: float = Field(0.0, description="TODO: document the initial maximum value.")


class SampleParams(PymatnextParams):
    """All parameters consumed by :func:`pymatnext.cli.sample.sample`."""

    general: GeneralParams = Field(
        default_factory=GeneralParams,
        description="TODO: document general nested-sampling controls.",
    )
    ns: NSParams = Field(..., description="TODO: document nested-sampling controls.")
    configs: ASEAtomsParams = Field(..., description="TODO: document configuration-generation controls.")

    @classmethod
    def default_data(cls):
        """Return the model-default layer in TOML-compatible key form."""

        return {"general": GeneralParams().model_dump()}


def load_sample_params(input_file, overrides):
    """Load parameter layers and return the validated top-level model.

    The precedence is Pydantic model defaults, package TOML, runtime TOML,
    then command-line overrides.
    """

    data = SampleParams.default_data()
    deep_update(data, load_packaged_toml("pymatnext", DEFAULTS_RESOURCE))

    with open(input_file) as fin:
        deep_update(data, decode_toml_none(toml.load(fin)))

    for spec in overrides:
        path, value = parse_override(spec)
        apply_override(data, path, value)

    return SampleParams.model_validate(data)


def format_sample_defaults():
    """Return the documented default TOML template for ``pymatnext-sample``."""

    return format_defaults(SampleParams, load_packaged_toml("pymatnext", DEFAULTS_RESOURCE))
