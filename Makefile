wheel:
	python3 -m pip wheel --no-deps .


pytest:
	pytest
	$(MAKE) pytest-mpi

pytest-mpi:
	mpirun -np 2 pytest --with-mpi
