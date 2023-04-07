from ase.calculators.morse import MorsePotential

calc = MorsePotential()

# consider switching to matscipy EAM if it's faster - broken for now, because calculate() calls
# aren't passing system_changes
#
# from pathlib import Path
# from matscipy.calculators.eam.calculator import EAM
# 
# calc = EAM(Path(__file__).parent / "../" / "do_EAM_LAMMPS" / "AlCu_Zhou04.eam.alloy", [1])
