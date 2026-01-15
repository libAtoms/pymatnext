import numpy as np

import ase.units

class ExitLoop():
    """Exit a loop when Z(T) at a particular temperature T is converged

    Parameters
    ----------
        ns: NS
            nested sampling object for params such as n_configs_global
        T: float
            temperature to converge down to
        T_to_ns_quant: float / "kB", default 1.0
            unit conversion to multiply temperature to get units of NS quantity, or string "kB" for ase.units.kB (K -> eV)
    """
    def __init__(self, *, ns, T, T_to_ns_quant=1.0):
        if T_to_ns_quant == "kB":
            T_to_ns_quant = ase.units.kB
        else:
            if not isinstance(T_to_ns_quant, float):
                raise ValueError(f"Invalid T_to_ns_quant={T_to_ns_quant}: neither float nor string 'kB'")

        self.beta = 1.0 / (T_to_ns_quant * T)
        self.log_a = np.log(ns.n_configs_global) - np.log(ns.n_configs_global + 1)
        self.log_1_minus_a = np.log(1.0 - np.exp(self.log_a))
        self.log_Z_term_max = np.NINF


    def __call__(self, loop_iter, NS_quant):
        """determine if loop should exit now based on current iteration and current nested sampling maximum quantity
        """
        # config_vol = a^loop_iter - a^(loop_iter + 1)
        #            = a^loop_iter (1 - a)
        log_config_vol = loop_iter * self.log_a + self.log_1_minus_a
        log_cur_Z_term = log_config_vol - self.beta * NS_quant
        self.log_Z_term_max = max(self.log_Z_term_max, log_cur_Z_term)
        # if this term contributed less that exp(-10) of max contribution, then Z(T) is converged and we can exit
        return log_cur_Z_term < self.log_Z_term_max - 10.0
