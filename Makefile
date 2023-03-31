wheel:
	python3 -m pip wheel --no-deps .


pytest:
	pytest
	$(MAKE) pytest-mpi

pytest-mpi:
	mpirun -np 2 pytest --with-mpi


PARAM_SRC = pymatnext/sample_params.py pymatnext/ns_params.py \
            pymatnext/loop_exit/loop_exit_params.py pymatnext/ns_configs/ase_atoms/ase_atoms_params.py
README_input_parameters.md: $(PARAM_SRC)
	./scripts/make_input_parameters_md > README_input_parameters.md
