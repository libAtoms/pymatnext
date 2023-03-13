import numpy as np

from pymatnext.ns_configs.ase_atoms.atoms_contig_store import AtomsContiguousStorage

def test_initial_create():
    at = AtomsContiguousStorage(symbols="H2", positions=[[0, 0, 0], [1, 0, 0]], cell = [5]*3, pbc=[True])
    assert at.is_contiguous

    # N 1 + symbols 2
    assert at.contig_storage_int.shape == (3,)
    # cell 9 + positions 6
    assert at.contig_storage_float.shape == (15,)

    np.allclose(at.cell, 5 * np.eye(3))
    np.allclose(at.positions, [[0, 0, 0], [1, 0, 0]])


def test_new_arrays():
    at = AtomsContiguousStorage(symbols="H2", positions=[[0, 0, 0], [1, 0, 0]], cell = [5]*3, pbc=[True])
    assert at.is_contiguous

    at.new_array("forces", np.ones((2,3)))
    assert not at.is_contiguous

    at.make_contiguous()
    assert at.is_contiguous

    # N 1 + symbols 2
    assert at.contig_storage_int.shape == (3,)
    # cell 9 + positions 6 + forces 6
    assert at.contig_storage_float.shape == (21,)

    np.allclose(at.cell, 5 * np.eye(3))
    np.allclose(at.positions, [[0, 0, 0], [1, 0, 0]])
    np.allclose(at.arrays["forces"], [[1, 1, 1], [1, 1, 1]])


def test_new_info():
    at = AtomsContiguousStorage(symbols="H2", positions=[[0, 0, 0], [1, 0, 0]], cell = [5]*3, pbc=[True])
    assert at.is_contiguous

    at.new_array("forces", np.ones((2,3)))
    assert not at.is_contiguous

    at.make_contiguous()
    assert at.is_contiguous

    at.info["energy"] = np.asarray(5.0)
    at.make_contiguous()

    np.allclose(at.cell, 5 * np.eye(3))
    np.allclose(at.positions, [[0, 0, 0], [1, 0, 0]])
    np.allclose(at.arrays["forces"], [[1, 1, 1], [1, 1, 1]])
    np.allclose(at.info["energy"], 5.0)

    at.info["energy"][...] = 4
    np.allclose(at.info["energy"], 4.0)
