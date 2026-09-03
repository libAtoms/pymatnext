"""Pydantic models for ``[configs]`` and ``[configs.walk]`` sections."""

from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import Field

from pymatnext.params import PymatnextParams


Composition = Union[str, List[Union[str, int]]]


class CalculatorParams(PymatnextParams):
    """Calculator selection and calculator-specific free-form arguments."""

    type: Annotated[str, Field(description="TODO: document the calculator type.")]
    args: Annotated[Dict[str, Any], Field(
        default_factory=dict,
        description="TODO: document calculator-specific keyword arguments.",
    )]


class MaxStepSizeParams(PymatnextParams):
    pos_gmc_each_atom: Annotated[float, Field(default=-0.1, description="TODO: document the GMC position step-size limit.")]
    cell_volume_per_atom: Annotated[float, Field(
        default=-0.05,
        description="TODO: document the cell-volume step-size limit per atom.",
    )]
    cell_shear_per_rt3_atom: Annotated[float, Field(
        default=-1.0,
        description="TODO: document the cell-shear step-size limit per atom.",
    )]
    cell_stretch: Annotated[float, Field(default=0.2, description="TODO: document the cell-stretch step-size limit.")]


class StepSizeParams(PymatnextParams):
    pos_gmc_each_atom: Annotated[float, Field(default=-1.0, description="TODO: document the GMC position step size.")]
    cell_volume_per_atom: Annotated[float, Field(
        default=-1.0,
        description="TODO: document the cell-volume step size per atom.",
    )]
    cell_shear_per_rt3_atom: Annotated[float, Field(
        default=-1.0,
        description="TODO: document the cell-shear step size per atom.",
    )]
    cell_stretch: Annotated[float, Field(default=-1.0, description="TODO: document the cell-stretch step size.")]


class SubmoveProbabilitiesParams(PymatnextParams):
    volume: Annotated[float, Field(default=0.7, description="TODO: document the relative probability of volume moves.")]
    shear: Annotated[float, Field(default=0.15, description="TODO: document the relative probability of shear moves.")]
    stretch: Annotated[float, Field(default=0.15, description="TODO: document the relative probability of stretch moves.")]


class CellWalkParams(PymatnextParams):
    min_aspect_ratio: Annotated[float, Field(default=0.8, description="TODO: document the minimum cell aspect ratio.")]
    flat_V_prior: Annotated[bool, Field(default=True, description="TODO: document the cell-volume prior.")]
    pressure_GPa: Annotated[float, Field(default=0.0, description="TODO: document the pressure in GPa.")]
    pressure: Annotated[Optional[float], Field(default=None, description="TODO: document the pressure in internal units.")]
    submove_probabilities: Annotated[SubmoveProbabilitiesParams, Field(
        default_factory=SubmoveProbabilitiesParams,
        description="TODO: document probabilities for individual cell submoves.",
    )]


class TypeWalkParams(PymatnextParams):
    sGC: Annotated[bool, Field(default=False, description="TODO: document semi-grand-canonical type moves.")]
    mu: Annotated[Dict[int, float], Field(
        default_factory=dict,
        description="TODO: document chemical potentials by atomic number.",
    )]


class ASEAtomsWalkParams(PymatnextParams):
    gmc_traj_len: Annotated[int, Field(default=8, description="TODO: document the GMC walk trajectory length.")]
    cell_traj_len: Annotated[int, Field(default=8, description="TODO: document the cell walk trajectory length.")]
    type_traj_len: Annotated[int, Field(default=8, description="TODO: document the type walk trajectory length.")]
    gmc_proportion: Annotated[float, Field(default=0.0, description="TODO: document the fraction of GMC moves.")]
    cell_proportion: Annotated[float, Field(default=0.0, description="TODO: document the fraction of cell moves.")]
    type_proportion: Annotated[float, Field(default=0.0, description="TODO: document the fraction of type moves.")]
    max_step_size: Annotated[MaxStepSizeParams, Field(
        default_factory=MaxStepSizeParams,
        description="TODO: document maximum accepted walk step sizes.",
    )]
    step_size: Annotated[StepSizeParams, Field(
        default_factory=StepSizeParams,
        description="TODO: document initial walk step sizes.",
    )]
    cell: Annotated[CellWalkParams, Field(
        default_factory=CellWalkParams,
        description="TODO: document cell-walk controls.",
    )]
    type: Annotated[TypeWalkParams, Field(
        default_factory=TypeWalkParams,
        description="TODO: document atom-type-walk controls.",
    )]
    combined: Annotated[bool, Field(default=False, description="TODO: document combined walk behavior.")]


class ASEAtomsParams(PymatnextParams):
    """Parameters for the built-in ASE Atoms nested-sampling configuration."""

    full_composition: Annotated[Composition, Field(default="", description="TODO: document the full system composition.")]
    composition: Annotated[Composition, Field(description="TODO: document the sampled composition.")]
    n_atoms: Annotated[int, Field(description="TODO: document the number of atoms in a configuration.")]
    dims: Annotated[int, Field(default=3, description="TODO: document the number of periodic dimensions.")]
    pbc: Annotated[List[bool], Field(
        default_factory=lambda: [True, True, True],
        description="TODO: document periodic boundary conditions by axis.",
    )]
    initial_rand_vol_per_atom: Annotated[float, Field(
        description="TODO: document the random initial volume per atom.",
    )]
    initial_rand_min_dist: Annotated[float, Field(
        description="TODO: document the minimum initial interatomic distance.",
    )]
    initial_rand_n_tries: Annotated[int, Field(
        default=10,
        description="TODO: document random-initialization retry attempts.",
    )]
    calculator: Annotated[CalculatorParams, Field(description="TODO: document calculator controls.")]
    walk: Annotated[ASEAtomsWalkParams, Field(description="TODO: document configuration walk controls.")]
    file: Annotated[Optional[str], Field(default=None, description="TODO: document an initial configuration file.")]
