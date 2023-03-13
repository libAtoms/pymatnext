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

        # preserve math for n_cull > 1, but not really supported here or elsewhere
        # NOTE: copied from pymatnest, but really not clear what it's doing (mathematically) - need to rederive and write better docs
        n_cull = 1
        # list of changes to effective numbers of walkers (for n_cull > 1)
        i_range_mod_n_cull = np.array(range(n_cull))
        # list of changes to effective numbers of walkers for next iter
        i_range_plus_1_mod_n_cull = np.mod(np.array(range(n_cull)) + 1, n_cull)
        # logs of relative compression ratios between subsequent culls
        log_X_n_term = np.log(ns.n_configs_global - i_range_mod_n_cull) - np.log(ns.n_configs_global + 1 - i_range_mod_n_cull)
        # cumulative sum of compression ratios over sequence of (multiple) culls 
        log_X_n_term_cumsum = np.cumsum(log_X_n_term)
        # cumulative sum 
        self.log_X_n_term_cumsum_modified = log_X_n_term_cumsum - np.log(ns.n_configs_global + 1 - i_range_plus_1_mod_n_cull)
        # overall sum
        self.log_X_n_term_sum = log_X_n_term_cumsum[-1]
        self.log_Z_term_max = np.NINF


    def __call__(self, loop_iter, max_val):
        """determine if loop should exit now based on current iteration and current nested sampling maximum quantity
        """
        # n_cull > 1 not supported here, so use the only element of log_a and max_val
        # copied from pymatnest, which refers to analyse.py calc_log_a() for the derivation of the math
        log_a = (self.log_X_n_term_sum * loop_iter + self.log_X_n_term_cumsum_modified)[0]
        self.log_Z_term_max = max(self.log_Z_term_max, np.amax(log_a - self.beta * max_val))
        log_Z_term_last = log_a - self.beta * max_val
        # if this term contributed less that exp(-10) of max contribution, then Z(T) is converged and we can exit
        return log_Z_term_last < self.log_Z_term_max - 10.0
