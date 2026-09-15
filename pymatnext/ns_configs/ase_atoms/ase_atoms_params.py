"""Pydantic models for ``[configs]`` and ``[configs.walk]`` sections."""

from typing import Annotated, Any, Union, Literal

from ase.units import GPa
from pydantic import AliasChoices, Field, PositiveInt, model_validator

from pymatnext.params import PymatnextParams


Composition = Union[str, list[Union[str, int]]]


class CalculatorParams(PymatnextParams):
    """Calculator selection and calculator-specific free-form arguments."""

    type: Annotated[Literal['ASE', 'LAMMPS'], Field(description="calculator type")]
    args: Annotated[dict[str, Any], Field(
        default_factory=dict,
        description="arbitrary args for calculator constructor. If 'ASE', 'module' with name of importable module defining "
                    "a `calc` Calculator object. If 'LAMMPS', 'cmds': list of `pair_style ...` etc. lammps commands, "
                    "'types': dict with atomic numbers or chemical symbols as keys and lammps types as values, "
                    "'header': optional list of header commands, 'cmd_args': optional command args, 'log_file': optional file for lammps log, "
                    "'name': optional lammps shared lib name, 'activate_mliappy_kokkos': optional bool for mliappy kokkos, "
                    "'boundary': optional args for lammps boundary command")]


class MaxStepSizeParams(PymatnextParams):
    pos_gmc_each_atom: Annotated[float, Field(default=-0.1, description="maximum GMC position move (for each atom), if negative multiplied by cube root of atomic volume")]
    cell_volume_per_atom: Annotated[float, Field(
        default=-0.05,
        description="maximum cell volume step (per atom), if negative multiplied by atomic volume")]
    cell_shear_per_rt3_atom: Annotated[float, Field(
        default=-1.0,
        description="maximum cell shear step (per natoms^1/3), if negative multiplied by cube root of atomic volume")]
    cell_stretch: Annotated[float, Field(default=0.2, description="maximum cell stretch (unitless strain)")]


class StepSizeParams(PymatnextParams):
    pos_gmc_each_atom: Annotated[float, Field(default=-1.0, description="initial GMC positions move (for each atom)")]
    cell_volume_per_atom: Annotated[float, Field(
        default=-1.0,
        description="initial cell volume step (per atom)",
    )]
    cell_shear_per_rt3_atom: Annotated[float, Field(
        default=-1.0,
        description="initial cell shear step (per natoms^1/3)",
    )]
    cell_stretch: Annotated[float, Field(default=-1.0, description="initial cell stretch step")]


class SubmoveProbabilitiesParams(PymatnextParams):
    volume: Annotated[float, Field(default=0.7, description="probability to try a cell volume move")]
    shear: Annotated[float, Field(default=0.15, description="probability to try a cell shear move")]
    stretch: Annotated[float, Field(default=0.15, description="probability to do a cell stretch move")]


class CellWalkParams(PymatnextParams):
    min_aspect_ratio: Annotated[float, Field(default=0.8, description="minimum cell aspect ratio to accept")]
    flat_V_prior: Annotated[bool, Field(default=True, description="use a prior independent of volume, rather than ensemble-correct V^natoms")]
    pressure: Annotated[float, Field(
        default=0.0,
        validation_alias=AliasChoices("pressure", "pressure_GPa"),
        description="applied pressure in eV/A^3 (or pressure_GPa in GPa)",
    )]
    submove_probabilities: Annotated[SubmoveProbabilitiesParams, Field(
        default_factory=SubmoveProbabilitiesParams,
        description="parameters controlling probabilities of volume, shear, and stretch move types",
    )]

    @model_validator(mode="before")
    @classmethod
    def _normalize_pressure(cls, data):
        if not isinstance(data, dict):
            return data
        if "pressure" in data and "pressure_GPa" in data:
            raise ValueError("Got both cell.pressure and cell.pressure_GPa")
        if "pressure_GPa" in data:
            data = data.copy()
            data["pressure"] = data.pop("pressure_GPa") * GPa
        return data


class TypeWalkParams(PymatnextParams):
    sGC: Annotated[bool, Field(default=False, description="semi-grand-canonical (species change) moves.")]
    mu: Annotated[dict[PositiveInt, float], Field(
        default_factory=dict,
        description="dict with atomic numbers as keys and chemical potentials as values for semi-grand-canonical moves",
    )]


class ASEAtomsWalkParams(PymatnextParams):
    gmc_traj_len: Annotated[int, Field(default=8, description="length of GMC walks")]
    cell_traj_len: Annotated[int, Field(default=8, description="length of cell move walks")]
    type_traj_len: Annotated[int, Field(default=8, description="length of type (sGC) move walks")]
    gmc_proportion: Annotated[float, Field(default=0.0, description="proportion of steps to do GMC walks with")]
    cell_proportion: Annotated[float, Field(default=0.0, description="proportion of steps to do cell walks with")]
    type_proportion: Annotated[float, Field(default=0.0, description="proportion of steps to do type (sGC) walks with")]
    max_step_size: Annotated[MaxStepSizeParams, Field(
        default_factory=MaxStepSizeParams,
        description="maximum step size for each move type",
    )]
    step_size: Annotated[StepSizeParams, Field(
        default_factory=StepSizeParams,
        description="initial step size for each move type",
    )]
    cell: Annotated[CellWalkParams, Field(
        default_factory=CellWalkParams,
        description="parameters controlling cell walks",
    )]
    type: Annotated[TypeWalkParams, Field(
        default_factory=TypeWalkParams,
        description="parameters controlling type (sGC) walks",
    )]
    combined: Annotated[bool, Field(default=False, description="use NS walk function that combines all move types")]


class ASEAtomsParams(PymatnextParams):
    """Parameters for the built-in ASE Atoms nested-sampling configuration."""

    full_composition: Annotated[Composition, Field(default="", description="composition that spans all possible elements that could appear")]
    composition: Annotated[Composition, Field(description="initial composition of configurations")]
    n_atoms: Annotated[int, Field(description="number of atoms in each configuration")]
    dims: Annotated[int, Field(default=3, description="number of dimensions (2 or 3)")]
    pbc: Annotated[list[bool], Field(
        default_factory=lambda: [True, True, True],
        description="periodicity of system along each cell vector",
    )]
    initial_rand_vol_per_atom: Annotated[float, Field(
        description="random initial configuration volume per atom",
    )]
    initial_rand_min_dist: Annotated[float, Field(
        description="random initial configuration minimum distance between atoms",
    )]
    initial_rand_n_tries: Annotated[int, Field(
        default=10,
        description="number of tries to get initial configuration that obeys minimum distances",
    )]
    calculator: Annotated[CalculatorParams, Field(description="parameters for calculator of interatomic interactions")]
    walk: Annotated[ASEAtomsWalkParams, Field(description="parameters for NS walks")]
