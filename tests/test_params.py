import pytest
from pydantic import ValidationError

from pymatnext.sample_params import SampleParams


def sample_data():
    return {
        "ns": {
            "n_walkers": 1,
            "walk_length": 1,
            "configs_module": "pymatnext.ns_configs.ase_atoms",
        },
        "configs": {
            "composition": "H",
            "n_atoms": 1,
            "initial_rand_vol_per_atom": 1.0,
            "initial_rand_min_dist": 0.5,
            "calculator": {"type": "ASE"},
            "walk": {"gmc_proportion": 1.0},
        },
    }


def test_model_fills_defaults():
    params = SampleParams.model_validate(sample_data())

    assert params.general.output_filename_prefix == "NS"
    assert params.ns.exit_conditions.module is None
    assert params.configs.walk.gmc_traj_len == 8


def test_model_requires_sections_and_rejects_unknown_fields():
    data = sample_data()
    del data["configs"]["walk"]
    with pytest.raises(ValidationError):
        SampleParams.model_validate(data)

    data = sample_data()
    data["configs"]["unknown"] = 1
    with pytest.raises(ValidationError):
        SampleParams.model_validate(data)

    data = sample_data()
    data["global"] = {"max_iter": 1}
    with pytest.raises(ValidationError):
        SampleParams.model_validate(data)
