"""Top-level Pydantic model for nested-sampling parameters."""

from typing import Annotated, Any, List

import toml
from pydantic import Field, PositiveInt, PositiveFloat, NonNegativeInt

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
    interval: Annotated[PositiveInt, Field(default=1000, description="NS iteration interval between step-size tuning")]
    n_configs: Annotated[PositiveInt, Field(default=1, description="number of configs to run when computing step size related statistics")]
    min_accept_rate: Annotated[PositiveFloat, Field(default=0.25, description="minimum accept rate for tuning step size")]
    max_accept_rate: Annotated[PositiveFloat, Field(default=0.5, description="maximum accept rate for tuning step size")]
    adjust_factor: Annotated[PositiveFloat, Field(default=1.25, description="TODO: document the tuning adjustment factor.")]


class WalkTrajectoryInfoParams(PymatnextParams):
    iter_min: Annotated[NonNegativeInt | None, Field(default=None, description="first iteration at which walk trajectory is saved")]
    iter_max: Annotated[int | None, Field(default=None, description="last iteration (inclusive) at which walk trajectory is saved, negative for no maximum")]
    interval: Annotated[PositiveInt | None, Field(default=None, description="interval at which walk trajectory is saved")]
    avg_times: Annotated[List[PositiveInt] | None, Field(default_factory=list, description="save walk trajectory with time averaging over these time scales")]


class GeneralParams(PymatnextParams):
    output_filename_prefix: Annotated[str, Field(default="NS", description="prefix for all output files")]
    output_filename_prefix_extra: Annotated[str, Field(default="", description="extra string to add to output_filename_prefix, designed for easy per-realization overriding")]
    random_seed: Annotated[PositiveInt, Field(description="seed for random number generator")]
    max_iter: Annotated[PositiveInt | None, Field(default=None, description="maximum NS iteration")]
    stdout_report_interval_s: Annotated[int, Field(default=60, description="interval in seconds between reports to stdout, < 0 for no reports")]
    sample_interval: Annotated[int, Field(default=1, description="interval in NS iterations between saved NS samples, <= 0 to disable")]
    traj_interval: Annotated[int, Field(default=100, description="interval in NS iterations between saved NS configurations, <= 0 to disable")]
    snapshot_interval: Annotated[int, Field(default=10000, description="interval in NS iterations between restart snapshots, <= 0 to disable")]
    snapshot_save_old: Annotated[int, Field(default=2, description="how many old snapshots so save")]
    step_size_tune: Annotated[StepSizeTuneParams, Field(
        default_factory=StepSizeTuneParams,
        description="parameters for tuning step sizes",
    )]
    walk_traj_info: Annotated[WalkTrajectoryInfoParams, Field(
        default_factory=WalkTrajectoryInfoParams,
        description="parametrs for saving walk trajectories",
    )]
    clone_history: Annotated[bool, Field(default=False, description="save a full history of which config was cloned at each iteration")]


class SampleParams(PymatnextParams):
    """All parameters consumed by :func:`pymatnext.cli.sample.sample`."""

    general: Annotated[GeneralParams, Field(
        default_factory=GeneralParams,
        description="General parameters",
    )]
    ns: Annotated[NSParams, Field(description="parameters for nested-sampling iteration process")]
    configs: Annotated[ASEAtomsParams, Field(description="config-type-specific parameters")]


def load_sample_params(input_file, overrides):
    """Load parameter layers and return the validated top-level model.

    Package TOML is merged with runtime TOML and command-line overrides;
    Pydantic field defaults fill any values that remain absent during final
    validation. The precedence is Pydantic defaults, package TOML, runtime
    TOML, then command-line overrides.
    """

    data = load_packaged_toml("pymatnext", DEFAULTS_RESOURCE)

    with open(input_file) as fin:
        deep_update(data, decode_toml_none(toml.load(fin)))

    for spec in overrides:
        path, value = parse_override(spec)
        apply_override(data, path, value)

    return SampleParams.model_validate(data)


def format_sample_defaults():
    """Return the documented default TOML template for ``pymatnext-sample``."""

    return format_defaults(SampleParams, load_packaged_toml("pymatnext", DEFAULTS_RESOURCE))
