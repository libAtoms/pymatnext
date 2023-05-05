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
def test_Morse_ASE_no_mpi(tmp_path, monkeypatch):
    do_Morse_ASE(tmp_path, monkeypatch, using_mpi=False)

@pytest.mark.mpi_skip
def test_Morse_ASE_restart_no_mpi(tmp_path, monkeypatch):
    do_Morse_ASE_restart(tmp_path, monkeypatch, using_mpi=False)

@pytest.mark.mpi_skip
def test_EAM_LAMMPS_no_mpi(tmp_path, monkeypatch):
    do_EAM_LAMMPS(tmp_path, monkeypatch, using_mpi=False, max_iter=300)

@pytest.mark.mpi_skip
def test_pressure_no_mpi(tmp_path, monkeypatch):
    do_pressure(tmp_path, monkeypatch, using_mpi=False)

@pytest.mark.mpi_skip
def test_sGC_no_mpi(tmp_path, monkeypatch):
    do_sGC(tmp_path, monkeypatch, using_mpi=False)

# tests with MPI

@pytest.mark.mpi
def test_Morse_ASE_mpi(mpi_tmp_path, monkeypatch):
    do_Morse_ASE(mpi_tmp_path, monkeypatch, using_mpi=True)

@pytest.mark.mpi
def test_Morse_ASE_restart_mpi(mpi_tmp_path, monkeypatch):
    do_Morse_ASE_restart(mpi_tmp_path, monkeypatch, using_mpi=True)

@pytest.mark.mpi
def test_EAM_LAMMPS_mpi(mpi_tmp_path, monkeypatch):
    do_EAM_LAMMPS(mpi_tmp_path, monkeypatch, using_mpi=True, max_iter=300)

@pytest.mark.mpi
def test_pressure_mpi(mpi_tmp_path, monkeypatch):
    do_pressure(mpi_tmp_path, monkeypatch, using_mpi=True)

@pytest.mark.mpi
def test_sGC_mpi(mpi_tmp_path, monkeypatch):
    do_sGC(mpi_tmp_path, monkeypatch, using_mpi=True)


def do_Morse_ASE_restart(tmp_path, monkeypatch, using_mpi):
    # run and stop at 75 iters, when the final line is not what the test expects
    try:
        do_Morse_ASE(tmp_path, monkeypatch, using_mpi=using_mpi, max_iter=75)
    except AssertionError:
        pass

    if using_mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0

    # add tag to .NS_samples header to detect overwriting vs. expected truncation + append
    assets_dir = Path(__file__).parent  / 'assets' / 'do_Morse_ASE'
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
    do_Morse_ASE(tmp_path, monkeypatch, using_mpi=using_mpi)

    # check that NS_samples file wasn't overwritten by looking for "restart" key in header
    with open(tmp_path / NS_samples_file) as fin:
        l = fin.readline()
    header = json.loads(l.replace('#', '', 1))
    assert header["restart"]


def do_Morse_ASE(tmp_path, monkeypatch, using_mpi, max_iter=None):
    if using_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 2:
            pytest.skip("do_Morse_ASE with MPI only works for comm.size == 2")

    assets_dir = Path(__file__).parent  / 'assets' / 'do_Morse_ASE'

    traj_interval = 100
    with open(assets_dir / 'params.toml') as fin, open(tmp_path / 'params.toml', 'w') as fout:
        for l in fin:
            # fix output_filename_prefix so everything is written to tmp_path
            if "output_filename_prefix" in l:
                fout.write(f'output_filename_prefix = "{tmp_path}/Morse_ASE"\n')
                continue

            # remember so correct file is checked
            if "traj_interval" in l:
                traj_interval = int(l.split()[2])

            # use as default for max_iter_use
            if "max_iter" in l:
                max_iter_params = int(l.split()[2])

            fout.write(l)

    if max_iter is None:
        max_iter_use = max_iter_params
    else:
        max_iter_use = max_iter

    # add assets dir for Morse.py  module
    sys.path.insert(0, str(assets_dir))
    if not using_mpi:
        # need PYMATNEXT_NO_MPI since otherwise rngs will vary depending on number of MPI processes, and 
        # numbers in output will vary
        monkeypatch.setenv("PYMATNEXT_NO_MPI", "1")

    main_args = ['--override_param', '/global/random_seed', '5', '--override_param', '/global/output_filename_prefix_extra', '.test',
                 '--override_param', '/global/clone_history', 'T',
                 str(tmp_path / 'params.toml')]
    if max_iter is not None:
        main_args = ['--override_param', '/global/max_iter', str(max_iter)] + main_args

    sample.main(main_args, mpi_finalize=False)
    del sys.path[0]

    # files exist
    assert (tmp_path / 'Morse_ASE.test.NS_samples').is_file()
    assert len(list(tmp_path.glob('Morse_ASE.test.traj.*xyz'))) == 1
    assert len(list(ase.io.read(tmp_path / 'Morse_ASE.test.traj.extxyz', ':'))) == max_iter_use // traj_interval

    # from test run 12/8/2022
    if using_mpi:
        fields_ref = np.asarray([9.90000000e+01, 8.04691943e+00, 1.28925862e+04, 1.60000000e+01])
    else:
        fields_ref = np.asarray([9.90000000e+01, 8.08011910e+00, 1.29457779e+04, 1.60000000e+01])

    with open(tmp_path / 'Morse_ASE.test.NS_samples') as fin:
        for l in fin:
            pass
    fields = np.asarray([float(f) for f in l.strip().split()])

    # tolerance loosened so that restart, which isn't perfect due to finite precision in
    # saved cofig file, still passes
    if not np.allclose(fields, fields_ref, rtol=0.02):
        print("final line test", fields)
        print("final line ref ", fields_ref)
        assert False

    # this will fail if number of steps is not divisible by N_samples interval
    with open(tmp_path / 'Morse_ASE.test.clone_history') as fin:
        N_clone_hist_lines = len(list(fin))
    assert N_clone_hist_lines == int(fields[0]) + 2


def do_EAM_LAMMPS(tmp_path, monkeypatch, using_mpi, max_iter=None):
    if lammps is None:
        pytest.skip("lammps module not installed")

    if using_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 2:
            pytest.skip("do_EAM_LAMMPS with MPI only works for comm.size == 2")

    assets_dir = Path(__file__).parent  / 'assets' / 'do_EAM_LAMMPS'

    traj_interval = 100
    with open(assets_dir / 'params_sGC.toml') as fin, open(tmp_path / 'params_sGC.toml', 'w') as fout:
        for l in fin:
            # fix output_filename_prefix so everything is written to tmp_path
            if "output_filename_prefix" in l:
                fout.write(f'output_filename_prefix = "{tmp_path}/EAM_LAMMPS"\n')
                continue

            # rewrite to find potentials in right place
            if "_POTENTIAL_DIR_" in l:
                fout.write(l.replace("_POTENTIAL_DIR_", str(assets_dir.resolve())))
                continue

            # use as default for max_iter_use
            if "max_iter" in l:
                max_iter_params = int(l.split()[2])

            # remember so correct file is checked
            if "traj_interval" in l:
                traj_interval = int(l.split()[2])

            fout.write(l)

    if max_iter is None:
        max_iter_use = max_iter_params
    else:
        max_iter_use = max_iter

    # add assets dir for Morse.py  module
    if not using_mpi:
        # need PYMATNEXT_NO_MPI since otherwise rngs will vary depending on number of MPI processes, and 
        # numbers in output will vary
        monkeypatch.setenv("PYMATNEXT_NO_MPI", "1")

    main_args = ['--override_param', '/global/random_seed', '5', '--override_param', '/global/output_filename_prefix_extra', '.test',
                 str(tmp_path / 'params_sGC.toml')]
    if max_iter is not None:
        main_args = ['--override_param', '/global/max_iter', str(max_iter)] + main_args

    sample.main(main_args, mpi_finalize=False)

    # files exist
    assert (tmp_path / 'EAM_LAMMPS.test.NS_samples').is_file()
    assert len(list(tmp_path.glob('EAM_LAMMPS.test.traj.*xyz'))) == 1
    assert len(list(ase.io.read(tmp_path / 'EAM_LAMMPS.test.traj.extxyz', ':'))) == max_iter_use // traj_interval

    # from test run 12/8/2022
    if using_mpi:
        fields_ref = np.asarray([2.99000000e+02, -3.91163426e+02,  1.08674253e+04,  1.60000000e+01, 1.87500000e-01,  8.12500000e-01])
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


def do_pressure(tmp_path, monkeypatch, using_mpi):
    if using_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 2:
            pytest.skip("do_pressure with MPI only works for comm.size == 2")

    assets_dir = Path(__file__).parent  / 'assets' / 'do_pressure'

    toml_files = ['params_ASE.toml']
    if lammps is not None:
        toml_files += ['params_LAMMPS.toml']

    for toml_file in toml_files:
        with open(assets_dir / toml_file) as fin, open(tmp_path / toml_file, 'w') as fout:
            for l in fin:
                # fix output_filename_prefix so everything is written to tmp_path
                if "output_filename_prefix" in l:
                    fout.write(f'output_filename_prefix = "{tmp_path}/{toml_file}"\n')
                    continue

                fout.write(l)

        # add assets dir for Morse.py  module
        sys.path.insert(0, str(assets_dir))
        # add assets dir for Morse.py  module
        if not using_mpi:
            # need PYMATNEXT_NO_MPI since otherwise rngs will vary depending on number of MPI processes, and 
            # numbers in output will vary
            monkeypatch.setenv("PYMATNEXT_NO_MPI", "1")

        main_args = ['--override_param', '/global/random_seed', '5', '--override_param', '/global/output_filename_prefix_extra', '.test',
                     str(tmp_path / toml_file)]

        sample.main(main_args, mpi_finalize=False)
        del sys.path[0]

        with open(tmp_path / (toml_file + ".test.NS_samples")) as fin:
            lfirst = None
            for l in fin:
                if l.startswith("#"):
                    continue

                if lfirst is None:
                    lfirst = l
            llast = l

        Vfirst = float(lfirst.strip().split()[2])
        Vlast = float(llast.strip().split()[2])
        assert Vfirst / Vlast > 20


def do_sGC(tmp_path, monkeypatch, using_mpi):
    if using_mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 2:
            pytest.skip("do_sGC with MPI only works for comm.size == 2")

    assets_dir = Path(__file__).parent  / 'assets' / 'do_sGC'

    toml_files = ['params_matscipy.toml']
    if lammps is not None:
        toml_files += ['params_LAMMPS.toml']

    for toml_file in toml_files:
        with open(assets_dir / toml_file) as fin, open(tmp_path / toml_file, 'w') as fout:
            for l in fin:
                # rewrite to find potentials in right place
                if "_POTENTIAL_DIR_" in l:
                    fout.write(l.replace("_POTENTIAL_DIR_", str(assets_dir.resolve())))
                    continue

                # fix output_filename_prefix so everything is written to tmp_path
                if "output_filename_prefix" in l:
                    fout.write(f'output_filename_prefix = "{tmp_path}/{toml_file}"\n')
                    continue

                fout.write(l)

        # add assets dir for Morse.py  module
        sys.path.insert(0, str(assets_dir))
        # add assets dir for Morse.py  module
        if not using_mpi:
            # need PYMATNEXT_NO_MPI since otherwise rngs will vary depending on number of MPI processes, and 
            # numbers in output will vary
            monkeypatch.setenv("PYMATNEXT_NO_MPI", "1")

        main_args = ['--override_param', '/global/random_seed', '5', '--override_param', '/global/output_filename_prefix_extra', '.test',
                     str(tmp_path / toml_file)]

        sample.main(main_args, mpi_finalize=False)
        del sys.path[0]

        with open(tmp_path / (toml_file + ".test.NS_samples")) as fin:
            lfirst = None
            for l in fin:
                if l.startswith("#"):
                    continue

                if lfirst is None:
                    lfirst = l
            llast = l

        f_13_first, f_29_first = [float(f) for f in lfirst.strip().split()[4:6]]
        f_13_last, f_29_last = [float(f) for f in llast.strip().split()[4:6]]

        assert f_29_first / f_13_first == 1.0
        assert f_29_last / f_13_last > 5
