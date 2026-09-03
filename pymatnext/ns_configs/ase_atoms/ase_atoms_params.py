"""Pydantic models for ``[configs]`` and ``[configs.walk]`` sections."""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from pymatnext.params import PymatnextParams


Composition = Union[str, List[Union[str, int]]]


class CalculatorParams(PymatnextParams):
    """Calculator selection and calculator-specific free-form arguments."""

    type: str
    args: Dict[str, Any] = Field(default_factory=dict)


class MaxStepSizeParams(PymatnextParams):
    pos_gmc_each_atom: float = -0.1
    cell_volume_per_atom: float = -0.05
    cell_shear_per_rt3_atom: float = -1.0
    cell_stretch: float = 0.2


class StepSizeParams(PymatnextParams):
    pos_gmc_each_atom: float = -1.0
    cell_volume_per_atom: float = -1.0
    cell_shear_per_rt3_atom: float = -1.0
    cell_stretch: float = -1.0


class SubmoveProbabilitiesParams(PymatnextParams):
    volume: float = 0.7
    shear: float = 0.15
    stretch: float = 0.15


class CellWalkParams(PymatnextParams):
    min_aspect_ratio: float = 0.8
    flat_V_prior: bool = True
    pressure_GPa: float = 0.0
    pressure: Optional[float] = None
    submove_probabilities: SubmoveProbabilitiesParams = Field(default_factory=SubmoveProbabilitiesParams)


class TypeWalkParams(PymatnextParams):
    sGC: bool = False
    mu: Dict[int, float] = Field(default_factory=dict)


class ASEAtomsWalkParams(PymatnextParams):
    gmc_traj_len: int = 8
    cell_traj_len: int = 8
    type_traj_len: int = 8
    gmc_proportion: float = 0.0
    cell_proportion: float = 0.0
    type_proportion: float = 0.0
    max_step_size: MaxStepSizeParams = Field(default_factory=MaxStepSizeParams)
    step_size: StepSizeParams = Field(default_factory=StepSizeParams)
    cell: CellWalkParams = Field(default_factory=CellWalkParams)
    type: TypeWalkParams = Field(default_factory=TypeWalkParams)
    combined: bool = False


class ASEAtomsParams(PymatnextParams):
    """Parameters for the built-in ASE Atoms nested-sampling configuration."""

    full_composition: Composition = ""
    composition: Composition
    n_atoms: int
    dims: int = 3
    pbc: List[bool] = Field(default_factory=lambda: [True, True, True])
    initial_rand_vol_per_atom: float
    initial_rand_min_dist: float
    initial_rand_n_tries: int = 10
    calculator: CalculatorParams
    walk: ASEAtomsWalkParams
    file: Optional[str] = None
