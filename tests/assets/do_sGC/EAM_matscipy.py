from pathlib import Path
from matscipy.calculators.eam.calculator import EAM

calc = EAM(Path(__file__).parent / "AlCu_Zhou04.eam.alloy", [13, 29])
