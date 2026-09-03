import textwrap

from pymatnext import sample_params


def test_load_sample_params_cli_overrides_input(tmp_path):
    params_file = tmp_path / "params.toml"
    params_file.write_text(textwrap.dedent("""\
        [general]
        max_iter = 4
        output_filename_prefix_extra = ".input"

        [ns]
        n_walkers = 1
        walk_length = 1
        configs_module = "pymatnext.ns_configs.ase_atoms"

        [configs]
        composition = "H"
        n_atoms = 1
        initial_rand_vol_per_atom = 1.0
        initial_rand_min_dist = 0.5

        [configs.calculator]
        type = "ASE"

        [configs.walk]
        gmc_proportion = 1.0
    """))
    params = sample_params.load_sample_params(
        params_file,
        ["general.max_iter=5", "general.output_filename_prefix_extra=\".override\""],
    )

    assert params.general.max_iter == 5
    assert params.general.output_filename_prefix_extra == ".override"
