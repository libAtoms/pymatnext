import pytest
from ase.units import GPa
from pydantic import ValidationError

from pymatnext.ns_configs.ase_atoms.ase_atoms_params import CellWalkParams
from pymatnext.sample_params import SampleParams


def sample_data():
    return {
        "general": {
            "random_seed": 5
        },
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
    assert params.ns.step_size_tune_walk_length is None
    assert params.ns.exit_conditions.module is None
    assert params.configs.walk.gmc_traj_len == 8


@pytest.mark.parametrize("value", [0, -1])
def test_step_size_tune_walk_length_must_be_positive(value):
    data = sample_data()
    data["ns"]["step_size_tune_walk_length"] = value

    with pytest.raises(ValidationError):
        SampleParams.model_validate(data)


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


def test_cell_walk_pressure_aliases_are_normalized():
    pressure = CellWalkParams.model_validate({"pressure": 0.25})
    pressure_gpa = CellWalkParams.model_validate({"pressure_GPa": 1.5})

    assert pressure.pressure == 0.25
    assert pressure_gpa.pressure == 1.5 * GPa
    assert "pressure_GPa" not in pressure_gpa.model_dump()


def test_cell_walk_pressure_aliases_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="Got both cell.pressure and cell.pressure_GPa"):
        CellWalkParams.model_validate({"pressure": 0.25, "pressure_GPa": 1.5})
