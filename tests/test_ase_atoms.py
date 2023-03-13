import numpy as np

from pymatnext.ns_configs.ase_atoms import NSConfig_ASE_Atoms

def test_composition():
    for composition in ["CuAl", "AlCu", ["Cu", "Al"], ["Al", "Cu"], [13, 29], [29, 13]]:
        Zs, counts = NSConfig_ASE_Atoms._parse_composition(composition)
        assert np.all(Zs == [13, 29])
        assert np.all(counts == [1, 1])

    for composition in ["Cu2Al", "AlCu2", ["Cu2", "Al"], ["Al", "Cu2"]]:
        Zs, counts = NSConfig_ASE_Atoms._parse_composition(composition)
        assert np.all(Zs == [13, 29])
        assert np.all(counts == [1, 2])


def test_initialize():
    params = { "composition": "Cu2Al" }
    NSConfig_ASE_Atoms.initialize(params)
    assert np.all(NSConfig_ASE_Atoms._Zs == [13, 29])
    assert NSConfig_ASE_Atoms.n_quantities == 3 + 2

    params = { "full_composition": "Cu2AlZn", "composition": "Cu2Al" }
    NSConfig_ASE_Atoms.initialize(params)
    assert np.all(NSConfig_ASE_Atoms._Zs == [13, 29, 30])
    assert NSConfig_ASE_Atoms.n_quantities == 3 + 3

def test_minimal():
    params = {"full_composition": "AlCuZn", "composition": "Cu2Al", "n_atoms": 9, "initial_rand_vol_per_atom": 10.0, "initial_rand_min_dist": 0.5,
              "calculator": {"type": "ASE"}, "walk": {"gmc_proportion": 1.0}}
    at = NSConfig_ASE_Atoms(params, allocate_only=True)
    assert np.all(at._Zs == [13, 29, 30])
    import ase.io, sys
    assert len(at.atoms) == 9
    assert sum(at.atoms.numbers == 13) == 3
    assert sum(at.atoms.numbers == 29) == 6
