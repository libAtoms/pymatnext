lammps.patch: new/*/*
	diff -N -u -r orig new > ../pymatnext/extras/lammps.patch || echo OK

cmake_lists:
	mkdir -p orig/cmake new/cmake
	cp ${LAMMPS_DIR}/cmake/CMakeLists.txt orig/cmake/
	python3 patch_cmake.py
