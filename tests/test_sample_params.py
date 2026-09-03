import textwrap

from pymatnext import sample_params


def test_load_sample_params_source_precedence(tmp_path, monkeypatch):
    params_file = tmp_path / "params.toml"
    params_file.write_text(textwrap.dedent("""\
        [global]
        max_iter = 4

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
    monkeypatch.setattr(
        sample_params,
        "load_packaged_toml",
        lambda package, resource: {"global": {"max_iter": 3}},
    )

    params = sample_params.load_sample_params(
        params_file,
        ["global.max_iter=5", "global.output_filename_prefix_extra=\".override\""],
    )

    assert params.global_.max_iter == 5
    assert params.global_.output_filename_prefix == "NS"
    assert params.global_.output_filename_prefix_extra == ".override"
    assert params.configs.walk.gmc_traj_len == 8
