import sys
import re
from pathlib import Path
import toml
import json

import numpy as np

import pytest

import ase.io

from pymatnext.cli import sample

try:
    import lammps
except:
    lammps = None

# tests without MPI

@pytest.mark.mpi_skip
def test_LJ_ASE_no_mpi(tmp_path, monkeypatch):
    do_LJ_ASE(tmp_path, monkeypatch, using_mpi=False)

@pytest.mark.mpi_skip
def test_LJ_ASE_restart_no_mpi(tmp_path, monkeypatch):
    do_LJ_ASE_restart(tmp_path, monkeypatch, using_mpi=False)

@pytest.mark.mpi_skip
def test_EAM_LAMMPS_no_mpi(tmp_path, monkeypatch):
    do_EAM_LAMMPS(tmp_path, monkeypatch, using_mpi=False, max_iter=300)

# tests with MPI

@pytest.mark.mpi
def test_LJ_ASE_mpi(mpi_tmp_path, monkeypatch):
    do_LJ_ASE(mpi_tmp_path, monkeypatch, using_mpi=True)

@pytest.mark.mpi
def test_LJ_ASE_restart_mpi(mpi_tmp_path, monkeypatch):
    do_LJ_ASE_restart(mpi_tmp_path, monkeypatch, using_mpi=True)

@pytest.mark.mpi
def test_EAM_LAMMPS_mpi(mpi_tmp_path, monkeypatch):
    do_EAM_LAMMPS(mpi_tmp_path, monkeypatch, using_mpi=True, max_iter=300)


def do_LJ_ASE_restart(tmp_path, monkeypatch, using_mpi):
    # run and stop at 75 iters, when the final line is not what the test expects
    try:
        do_LJ_ASE(tmp_path, monkeypatch, using_mpi=using_mpi, max_iter=75)
    except AssertionError:
        pass

    if using_mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0

    # add tag to .NS_samples header to detect overwriting vs. expected truncation + append
    assets_dir = Path(__file__).parent  / 'assets' / 'example_LJ_ASE'
    with open(assets_dir / 'params.toml') as fin:
        params = toml.load(fin)
    output_filename_prefix = params["global"]["output_filename_prefix"]
    NS_samples_file = f'{output_filename_prefix}.test.NS_samples'

    if rank == 0:
        with open(tmp_path / NS_samples_file) as fin, open(tmp_path / (NS_samples_file + '.new'), 'w') as fout:
            l = fin.readline().strip()
            l = re.sub(r'}\s*$', ', "restart": true}', l)
            fout.write(l + '\n')
            for l in fin:
                fout.write(l)
        (tmp_path / (NS_samples_file + '.new')).rename(tmp_path / NS_samples_file)

    # run again, should restart from existing files
    do_LJ_ASE(tmp_path, monkeypatch, using_mpi=using_mpi)

    # check that NS_samples file wasn't overwritten by looking for "restart" key in header
    with open(tmp_path / NS_samples_file) as fin:
        l = fin.readline()
    header = json.loads(l.replace('#', '', 1))
    assert header["restart"]


def do_LJ_ASE(tmp_path, monkeypatch, using_mpi, max_iter=None):
    if using_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 2:
            pytest.skip("do_LJ_ASE with MPI only works for comm.size == 2")

    assets_dir = Path(__file__).parent  / 'assets' / 'example_LJ_ASE'

    traj_interval = 100
    with open(assets_dir / 'params.toml') as fin, open(tmp_path / 'params.toml', 'w') as fout:
        for l in fin:
            if "output_filename_prefix" in l:
                fout.write(f'output_filename_prefix = "{tmp_path}/LJ_ASE"\n')
                continue

            if "traj_interval" in l:
                traj_interval = int(l.split()[2])

            if "max_iter" in l:
                max_iter_params = int(l.split()[2])

            fout.write(l)

    if max_iter is None:
        max_iter_use = max_iter_params
    else:
        max_iter_use = max_iter

    # add assets dir for LJ.py  module
    sys.path.insert(0, assets_dir)
    if not using_mpi:
        # need PYMATNEXT_NO_MPI since otherwise rngs will vary depending on number of MPI processes, and 
        # numbers in output will vary
        monkeypatch.setenv("PYMATNEXT_NO_MPI", "1")

    main_args = ['--random_seed', '5', '--output_filename_postfix', '.test',
                 str(tmp_path / 'params.toml')]
    if max_iter is not None:
        main_args = ['--max_iter', str(max_iter)] + main_args

    sample.main(main_args, mpi_finalize=False)
    del sys.path[0]

    # files exist
    assert (tmp_path / 'LJ_ASE.test.NS_samples').is_file()
    assert len(list(tmp_path.glob('LJ_ASE.test.traj.*xyz'))) == 1
    assert len(list(ase.io.read(tmp_path / 'LJ_ASE.test.traj.extxyz', ':'))) == max_iter_use // traj_interval

    # from test run 12/8/2022
    if using_mpi:
        fields_ref = np.asarray([9.90000000e+01, 8.07238033e+00, 1.29333790e+04, 1.60000000e+01])
    else:
        fields_ref = np.asarray([9.90000000e+01, 8.04772245e+00, 1.28972604e+04, 1.60000000e+01])

    with open(tmp_path / 'LJ_ASE.test.NS_samples') as fin:
        for l in fin:
            pass
    fields = np.asarray([float(f) for f in l.strip().split()])

    if not np.allclose(fields, fields_ref):
        print("final line test", fields)
        print("final line ref ", fields_ref)
        assert False


@pytest.mark.skipif(lammps is None)
def do_EAM_LAMMPS(tmp_path, monkeypatch, using_mpi, max_iter=None):
    if using_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 2:
            pytest.skip("do_EAM_LAMMPS with MPI only works for comm.size == 2")

    print("BOB running in", tmp_path)

    assets_dir = Path(__file__).parent  / 'assets' / 'example_EAM_LAMMPS'

    traj_interval = 100
    with open(assets_dir / 'params_sGC.toml') as fin, open(tmp_path / 'params_sGC.toml', 'w') as fout:
        for l in fin:
            if "output_filename_prefix" in l:
                fout.write(f'output_filename_prefix = "{tmp_path}/EAM_LAMMPS"\n')
                continue

            if "_POTENTIAL_DIR_" in l:
                fout.write(l.replace("_POTENTIAL_DIR_", str(assets_dir.resolve())))
                continue

            if "max_iter" in l:
                max_iter_file = int(l.split()[2])

            if "traj_interval" in l:
                traj_interval = int(l.split()[2])

            fout.write(l)

    if max_iter is None:
        max_iter_use = max_iter_params
    else:
        max_iter_use = max_iter

    # add assets dir for LJ.py  module
    if not using_mpi:
        # need PYMATNEXT_NO_MPI since otherwise rngs will vary depending on number of MPI processes, and 
        # numbers in output will vary
        monkeypatch.setenv("PYMATNEXT_NO_MPI", "1")

    main_args = ['--random_seed', '5', '--output_filename_postfix', '.test',
                 str(tmp_path / 'params_sGC.toml')]
    if max_iter is not None:
        main_args = ['--max_iter', str(max_iter)] + main_args

    sample.main(main_args, mpi_finalize=False)
    del sys.path[0]

    # files exist
    assert (tmp_path / 'EAM_LAMMPS.test.NS_samples').is_file()
    assert len(list(tmp_path.glob('EAM_LAMMPS.test.traj.*xyz'))) == 1
    assert len(list(ase.io.read(tmp_path / 'EAM_LAMMPS.test.traj.extxyz', ':'))) == max_iter_use // traj_interval

    # from test run 12/8/2022
    if using_mpi:
        fields_ref = np.asarray([299, -391.4317494342, 22269.1573663966, 16.0000000000, 0.1875000000, 0.8125000000])
    else:
        fields_ref = np.asarray([299, -366.1823642208, 6004.3693892916, 16.0000000000, 0.2500000000, 0.7500000000])

    with open(tmp_path / 'EAM_LAMMPS.test.NS_samples') as fin:
        for l in fin:
            pass
    fields = np.asarray([float(f) for f in l.strip().split()])

    if not np.allclose(fields, fields_ref):
        print("final line test", fields)
        print("final line ref ", fields_ref)
        assert False
