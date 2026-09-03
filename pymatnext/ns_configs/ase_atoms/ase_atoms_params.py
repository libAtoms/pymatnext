"""Pydantic models for ``[configs]`` and ``[configs.walk]`` sections."""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from pymatnext.params import PymatnextParams


Composition = Union[str, List[Union[str, int]]]


class CalculatorParams(PymatnextParams):
    """Calculator selection and calculator-specific free-form arguments."""

    type: str = Field(..., description="TODO: document the calculator type.")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="TODO: document calculator-specific keyword arguments.",
    )


class MaxStepSizeParams(PymatnextParams):
    pos_gmc_each_atom: float = Field(-0.1, description="TODO: document the GMC position step-size limit.")
    cell_volume_per_atom: float = Field(
        -0.05,
        description="TODO: document the cell-volume step-size limit per atom.",
    )
    cell_shear_per_rt3_atom: float = Field(
        -1.0,
        description="TODO: document the cell-shear step-size limit per atom.",
    )
    cell_stretch: float = Field(0.2, description="TODO: document the cell-stretch step-size limit.")


class StepSizeParams(PymatnextParams):
    pos_gmc_each_atom: float = Field(-1.0, description="TODO: document the GMC position step size.")
    cell_volume_per_atom: float = Field(
        -1.0,
        description="TODO: document the cell-volume step size per atom.",
    )
    cell_shear_per_rt3_atom: float = Field(
        -1.0,
        description="TODO: document the cell-shear step size per atom.",
    )
    cell_stretch: float = Field(-1.0, description="TODO: document the cell-stretch step size.")


class SubmoveProbabilitiesParams(PymatnextParams):
    volume: float = Field(0.7, description="TODO: document the relative probability of volume moves.")
    shear: float = Field(0.15, description="TODO: document the relative probability of shear moves.")
    stretch: float = Field(0.15, description="TODO: document the relative probability of stretch moves.")


class CellWalkParams(PymatnextParams):
    min_aspect_ratio: float = Field(0.8, description="TODO: document the minimum cell aspect ratio.")
    flat_V_prior: bool = Field(True, description="TODO: document the cell-volume prior.")
    pressure_GPa: float = Field(0.0, description="TODO: document the pressure in GPa.")
    pressure: Optional[float] = Field(None, description="TODO: document the pressure in internal units.")
    submove_probabilities: SubmoveProbabilitiesParams = Field(
        default_factory=SubmoveProbabilitiesParams,
        description="TODO: document probabilities for individual cell submoves.",
    )


class TypeWalkParams(PymatnextParams):
    sGC: bool = Field(False, description="TODO: document semi-grand-canonical type moves.")
    mu: Dict[int, float] = Field(
        default_factory=dict,
        description="TODO: document chemical potentials by atomic number.",
    )


class ASEAtomsWalkParams(PymatnextParams):
    gmc_traj_len: int = Field(8, description="TODO: document the GMC walk trajectory length.")
    cell_traj_len: int = Field(8, description="TODO: document the cell walk trajectory length.")
    type_traj_len: int = Field(8, description="TODO: document the type walk trajectory length.")
    gmc_proportion: float = Field(0.0, description="TODO: document the fraction of GMC moves.")
    cell_proportion: float = Field(0.0, description="TODO: document the fraction of cell moves.")
    type_proportion: float = Field(0.0, description="TODO: document the fraction of type moves.")
    max_step_size: MaxStepSizeParams = Field(
        default_factory=MaxStepSizeParams,
        description="TODO: document maximum accepted walk step sizes.",
    )
    step_size: StepSizeParams = Field(
        default_factory=StepSizeParams,
        description="TODO: document initial walk step sizes.",
    )
    cell: CellWalkParams = Field(
        default_factory=CellWalkParams,
        description="TODO: document cell-walk controls.",
    )
    type: TypeWalkParams = Field(
        default_factory=TypeWalkParams,
        description="TODO: document atom-type-walk controls.",
    )
    combined: bool = Field(False, description="TODO: document combined walk behavior.")


class ASEAtomsParams(PymatnextParams):
    """Parameters for the built-in ASE Atoms nested-sampling configuration."""

    full_composition: Composition = Field("", description="TODO: document the full system composition.")
    composition: Composition = Field(..., description="TODO: document the sampled composition.")
    n_atoms: int = Field(..., description="TODO: document the number of atoms in a configuration.")
    dims: int = Field(3, description="TODO: document the number of periodic dimensions.")
    pbc: List[bool] = Field(
        default_factory=lambda: [True, True, True],
        description="TODO: document periodic boundary conditions by axis.",
    )
    initial_rand_vol_per_atom: float = Field(
        ...,
        description="TODO: document the random initial volume per atom.",
    )
    initial_rand_min_dist: float = Field(
        ...,
        description="TODO: document the minimum initial interatomic distance.",
    )
    initial_rand_n_tries: int = Field(
        10,
        description="TODO: document random-initialization retry attempts.",
    )
    calculator: CalculatorParams = Field(..., description="TODO: document calculator controls.")
    walk: ASEAtomsWalkParams = Field(..., description="TODO: document configuration walk controls.")
    file: Optional[str] = Field(None, description="TODO: document an initial configuration file.")
