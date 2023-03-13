import numpy as np

from ase.atoms import Atoms

class AtomsContiguousStorage(Atoms):
    """Extended ``Atoms`` class with _contiguous_ storage of number of atoms, ``Atoms.cell``, 
    and any item in ``Atoms.arrays`` and ``Atoms.info`` that is an np.ndarray with dtype np.int64
    or np.float64.
    
    Useful for more efficient MPI communication

    Note that behavior is somewhat fragile.  In particular, it's easy to overwrite the
    values of arrays and info dicts in a way that makes them no longer mapped to the
    contiguous storage, e.g. starting from an ``atoms.info["energy"]`` that is a
    0-dim ndarray view into the contiguous storage, ``atoms.info["energy"] = 5.0``
    breaks this mapping.  Instead you must do ``atoms.info["energy"][...] = 5.0``

    Instances have contig_storage_int np.ndarray(dtype=np.int64) and
    contig_storage_float np.ndarray(dtype=np.float64) attributes

    Parameters
    ----------
    from_atoms: Atoms, optional
        source object to create this object from. No other Atoms constructor
        arguments are allowed if this is present
    *args, **kwargs: various
        Atoms() constructor positional and keyword args
    """
    def __init__(self, *args, **kwargs):
        if "from_atoms" in kwargs:
            # copy constructor
            assert len(args) == 0 and len(kwargs) == 1
            assert isinstance(kwargs["from_atoms"], Atoms)
            source = kwargs["from_atoms"].copy()
            super().__init__()
            for attr in dir(source):
                setattr(self, attr, getattr(source, attr))
        else:
            super().__init__(*args, **kwargs)
        self.make_contiguous()


    def new_array(self, *args, **kwargs):
        """Add a new array and set is_contiguous to false

        See also Atoms.new_array()

        Parameters
        ----------
        *args, **kwargs: various
            Atoms.new_array() positional and keyword arguments
        """

        super().new_array(*args, **kwargs)
        self.is_contiguous = False


    def make_contiguous(self):
        """Make storage contiguous

        Data is stored in self.contig_storage_int and self.contig_float_storage
        """

        n_ints, n_floats = self._count_info_arrays_storage()
        n_ints += 1 # number of atoms
        n_floats += 9 # cell

        self.contig_storage_int = np.zeros(n_ints, dtype=np.int64)
        self.contig_storage_float = np.zeros(n_floats, dtype=np.float64)

        # store number of atoms
        self.contig_storage_int[0] = len(self)
        int_offset = 1

        # cell is not in arrays, store separately
        float_offset = 0
        float_offset = self._cell_use_prealloc(float_offset)

        # store all ndarrays in info and arrays
        int_offset, float_offset = self._set_info_arrays_storage_as_views(int_offset, float_offset, copy=True)

        self.is_contiguous = True


    def _cell_use_prealloc(self, float_offset):
        """Store ase.cell.Cell data in a preallocated array

        Parameters
        ----------
        float_offset: int
            offset into current position in self.contig_storage_float

        Returns
        -------
        new_float_offset: int
            new value of float offset after the newly inserted data
        """

        self.contig_storage_float[float_offset:float_offset + 9] = self.cell.reshape((-1))
        self.cell.array = self.contig_storage_float[0:9].reshape((3,3))

        return float_offset + 9


    def _count_info_arrays_storage(self):
        """count number of integers and floats in atoms.info and atoms.arrays for storage
        """
        n_ints = 0
        n_floats = 0
        for data_dict in [self.arrays, self.info]:
            for item_name in data_dict:
                item = data_dict[item_name]
                # create correct shape view into preallocated space
                if not isinstance(item, np.ndarray):
                    continue

                if item.dtype == np.int64:
                    n_ints += item.size
                elif item.dtype == np.float64:
                    n_floats += item.size

        return n_ints, n_floats


    def _set_info_arrays_storage_as_views(self, int_offset, float_offset, copy):
        """set self.info and self.arrays to views inside self.contig_storage_int and self.contig_storage_float
        """

        for data_dict in [self.arrays, self.info]:
            for item_name in data_dict:
                item = data_dict[item_name]
                # create correct shape view into preallocated space
                if not isinstance(item, np.ndarray):
                    continue

                prealloc_view = None
                if item.dtype == np.int64:
                    prealloc_view = self.contig_storage_int[int_offset:int_offset + item.size].reshape(item.shape)
                    int_offset += item.size
                elif item.dtype == np.float64:
                    prealloc_view = self.contig_storage_float[float_offset:float_offset + item.size].reshape(item.shape)
                    float_offset += item.size

                if prealloc_view is not None:
                    if copy:
                        # copy data into preallocated space
                        prealloc_view[...] = item
                    # make array dict entry point to preallocated space
                    data_dict[item_name] = prealloc_view

        return int_offset, float_offset


# NOT REALLY NEEDED, AND MAY REQUIRE FEATURES THAT ARE NO LONGER IMPLEMENTED
#
# from ase.atom import Atom
# 
# def atoms_from_contiguous(from_int, from_float, to_atoms):
#     """Copy from int and float preallocated arrays into existing Atoms object
# 
#     Parameters:
# 
#     from_int: np.ndarray dtype=int64
#         contiguous integer data
# 
#     from_float: np.ndarray dtype=float64
#         contiguous float data
# 
#     to_atoms: Atoms
#         atoms object to set data of
#     """
#     from_N = from_int[0]
#     to_N = len(to_atoms)
# 
#     if from_N == to_N and getattr(to_atoms, 'is_contiguous', False):
#         # same number of atoms: assume same fields (check overall size) and just copy data
#         assert from_int.shape == to_atoms._int_space.shape
#         assert from_float.shape == to_atoms._float_space.shape
# 
#         to_atoms._int_space[:] = from_int
#         to_atoms._float_space[:] = from_float
#     else:
#         if to_N < from_N:
#             # add
#             to_atoms += Atoms([Atom()] * (from_N - to_N))
#         else:
#             # to_N > from_N, delete
#             del to_atoms[0:(to_N - from_N)]
# 
#         assert from_N == len(to_atoms)
# 
#         # use existing storage
#         to_atoms._int_space = from_int
#         to_atoms._float_space = from_float
# 
#         _cell_use_prealloc(to_atoms.cell, to_atoms._float_space[0:9].reshape((3,3)), copy=False)
# 
#         # different dimensions, have to copy manually, but still assume same fields
#         int_offset = 1
#         float_offset = 9
#         _set_arrays_views(to_atoms, int_offset, float_offset, copy=False)
# 
#     to_atoms.is_contiguous = True
