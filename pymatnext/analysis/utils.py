import numpy as np

def calc_log_a(iters, n_walkers, n_cull, each_cull=False):
    if each_cull:
        # assume that for multiple culls, every energy is reported, use formula from
        #     SENS paper PRX v. 4 p 031034 (2014) Eq. 3
        # also assume that iters array increments by one for each cull (i.e. not exactly NS iters)
        # X_n = \prod_{i=0}^n \frac{N-i\%P}{N+1-i\%P}
        # \log X_n = \sum_{i=0}^n \log (N-i\%P) - \log(N+1-i\%P)
        ## using leading underscore to suppress ruff for this unused bit of code
        _i_range_mod_n_cull = np.array(range(0, iters[-1] + 1)) % n_cull
        _log_X_n_term = np.log(n_walkers - _i_range_mod_n_cull) - np.log(n_walkers + 1 - _i_range_mod_n_cull)
        _log_X_n = np.cumsum(_log_X_n_term)
        # a(iter[i]) = X(iter[i-1]) - X(iter[i])
        #     = prod(0..iter[i-1]) (N-i%P)/(N+1-i%P) - prod(0..iter[i]) (N-i%P)/(N+1-i%P)
        #     = [ prod(0..iter[i-1]) (N-i%P)/(N+1-i%P) ] * (1 - prod(iter[i-1]+1..iter[i]) (N-i%P)/(N+1-i%P))
        #     = [ prod(0..iter[i-1]) (N-i%P)/(N+1-i%P) ] * (1 - prod(iter[i-1]+1..iter[i]) (N-i%P)/(N+1-i%P))
        raise RuntimeError('calc_log_a for each_cull not yet implemented')
    else:
        log_a = iters * np.log((n_walkers - n_cull + 1) / (n_walkers + 1))

    return log_a


def calc_Z_terms(beta, log_a, Es, flat_V_prior=False, N_atoms=None, Vs=None):
    """Return the terms that sum to Z

    Parameters
    ----------
    beta: float
        1/(kB T)
    log_a: list(float)
        log of NS factors
    Es: list(float)
        energies
    flat_V_prior: bool, default False
        data came from flat V prior NS, needs to be reweighted by V^N_atoms
    N_atoms: int / list(int), default None
        number of atoms, needed for flat_V_prior
    Vs: list(float)
        volume of cell, needed for flat_V_prio

    Returns
    -------
    Z_term: list of terms that sum to Z, multipled by exp(-shift)
    log_shift: shift subtracted from each log(Z_term_true) to get log(Z_term)
    """
    log_Z_term = log_a[:] - beta * Es[:]

    if flat_V_prior:
        if N_atoms is None or Vs is None:
            raise RuntimeError('flat_V_prior requires numbers of atoms and volumes to reweight')
        log_Z_term += N_atoms * np.log(Vs[:])

    log_shift = np.amax(log_Z_term[:])
    Z_term = np.exp(log_Z_term[:] - log_shift)

    return (Z_term, log_shift)


def analyse_T(T, Es, E_shift, Vs, extra_vals, log_a, flat_V_prior, N_atoms, kB, n_extra_DOF, p_entropy_min=5.0, sum_f=np.sum):
    """Do an analysis at a single temperature

    Note that some parameters (e.g. percentiles used for low and high extent of the contributing iterations,
    amount of clipping of distribution at large iteration that indicates a problem) should probably be optional
    arguments

    Parameters
    ----------
    T: float
        temperature
    Es: ndarray(float)
        energies at each iter
    E_shift: float
        value that was subtracted from Es
    Vs: ndarray(float), optional
        volumes at each iter, required if flat_V_prior is true
    extra_vals: list(ndarray(float)), optional
        extra values to average according to the usual weights
        list of ndarrays, each with last dimension equal to len(Es)
    log_a: list(float)
        compression factor for each term, same len as Es
    flat_V_prior: bool
        if true, sample flat in V, otherwise sample probability includes V^Natoms factor
    N_atoms: ndarray(int)
        numbers of atoms, required if flat_V_prior is True
    kB: float
        Boltzmann constant to convert between T and E
    n_extra_DOF: int
        number of extra degrees of freedom per atom that aren't included in Es (e.g. kinetic), to be
        added analytically to energies and specific heats
    p_entropy_min: float, default 5
        minimum value of entropy of probability distribution that indicates a problem (poor sampling, e.g. with P reweighting)

    Returns
    -------
    dict of enesmble averages of various thermodynamic quantities and extra_vals
    """
    beta = 1.0 / (kB * T)

    # Z_term here is actually Z_term_true * exp(-log_shift)
    (Z_term, log_shift) = calc_Z_terms(beta, log_a, Es, flat_V_prior, N_atoms, Vs)

    # Note that
    #     Z_term = Z_term_true * exp(-log_shift)
    # exp(-log_shift) constant factor doesn't matter for quantities that are calculated from sums
    # weighted with Z_term and normalized by Z_term_sum
    Z_term_sum = sum_f(Z_term)

    U_pot = sum_f(Z_term * Es) / Z_term_sum

    if N_atoms is not None:
        N = sum_f(Z_term * N_atoms) / Z_term_sum
        n_extra_DOF * N

    U = n_extra_DOF / (2.0 * beta) + U_pot + E_shift

    Cvp = n_extra_DOF * kB / 2.0 + kB * beta * beta * (sum(Z_term * Es**2) / Z_term_sum - U_pot**2)

    if Vs is not None:
        V = sum_f(Z_term * Vs) / Z_term_sum
        thermal_exp = -1.0 / V * kB * beta * beta * (sum(Z_term * Vs) * sum(Z_term * Es) / Z_term_sum - sum(Z_term * Vs * Es)) / Z_term_sum
    else:
        V = None
        thermal_exp = None

    if extra_vals is not None and len(extra_vals) > 0:
        extra_vals_out = []
        for v in (extra_vals):
            extra_vals_out.append(sum_f(Z_term * v, axis=-1) / Z_term_sum)

    # undo shift of Z_term
    log_Z = np.log(Z_term_sum) + log_shift

    # we want last Z term to have w = 1, so we define a factor f which scales it correctly
    #    f exp(log_shift) Z_term[-1] = 1.0 * exp(-beta Es[-1])
    #    f = exp(-beta Es[-1] - log_shift) / Z_term[-1]
    #    log(f) = -beta Es[-1] - log_shift - log(Z_term[-1])
    log_f = -beta * Es[-1] - log_shift - np.log(Z_term[-1])
    # this factor rescales every term in Z
    log_Z += log_f

    # also add the E_shift
    Helmholtz_F = -log_Z / beta + E_shift

    mode_config = np.argmax(Z_term)


    results_dict = {'log_Z': log_Z,
                    'FG': Helmholtz_F,
                    'U': U,
                    'S': (U - Helmholtz_F) * beta,
                    'Cvp': Cvp}

    if Vs is not None:
        results_dict['V'] = V
        results_dict['thermal_exp'] = thermal_exp

    # compute range of configs that contributes significantly to sum
    Z_term_cumsum = np.cumsum(Z_term)
    low_percentile_config = np.where(Z_term_cumsum < 0.01 * Z_term_sum)[0][-1]
    high_percentile_config = np.where(Z_term_cumsum > 0.99 * Z_term_sum)[0][0] + 1
    high_percentile_config = min(high_percentile_config, len(Z_term) - 1)

    probabilities = Z_term / Z_term_sum
    probabilities = probabilities[np.where(probabilities > 0.0)]
    p_entropy = -sum_f(probabilities * np.log(probabilities))

    results_dict.update({'low_percentile_config': low_percentile_config,
                         'mode_config': mode_config,
                         'high_percentile_config': high_percentile_config,
                         'p_entropy': p_entropy})

    if extra_vals is not None and len(extra_vals) > 0:
        results_dict['extra_vals'] = extra_vals_out

    # Finally, check for sampling problems
    problem = False
    # one way to get bad sampling is to be too dominated by a few configurations
    problem |= p_entropy < p_entropy_min
    # another is to clip the top (high iteration #) of the distribution
    low_percentile_mean = np.mean(Z_term[low_percentile_config:low_percentile_config + 1000] / Z_term_sum)
    high_percentile_mean = np.mean(Z_term[high_percentile_config - 1000:high_percentile_config] / Z_term_sum)
    problem |= high_percentile_mean / low_percentile_mean > 2.0

    results_dict['problem'] = 'true' if problem else 'false'

    return results_dict
