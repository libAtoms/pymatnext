lammps.patch: new/*/* cmake_lists
	diff -N -u -r orig new > ../pymatnext/extras/lammps.patch || echo OK

cmake_lists:
	mkdir -p orig/cmake new/cmake
	if [ -z ${PRISTINE_LAMMPS_DIR} ]; then wget https://raw.githubusercontent.com/lammps/lammps/develop/cmake/CMakeLists.txt; mv CMakeLists.txt orig/cmake/; else echo "Using PRISTINE_LAMMPS_DIR ${PRISTINE_LAMMPS_DIR}" 1>&2 ; cp ${PRISTINE_LAMMPS_DIR}/cmake/CMakeLists.txt orig/cmake/; fi
	python3 patch_cmake.py
