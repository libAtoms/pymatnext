import logging
import numpy as np

def calc_log_a(iters, n_walkers, n_cull, discrete=False, n_beta_samples=0, beta_seed=2845435):
    if discrete:
        iters = np.asarray(iters)
        n_cull = np.asarray(n_cull)
        # need every iter, and the number culled for each iter
        assert len(iters) == len(n_cull)
        assert np.all(iters[1:] - iters[:-1] == 1)
        # fraction remaining after each iteration
        fracs = (n_walkers - n_cull) / n_walkers
        pathological_iters = np.where(fracs == 0)[0]
        if len(pathological_iters) != 0:
            logging.warning(f"Found fraction culled = 1 at iters {pathological_iters}. "
                            "Sign of underconvergence w.r.t. number of walkers")
            fracs[pathological_iters] = 1.0 / (n_walkers + 1.0)

        # volume remaining after iteration i
        # vol_i = \prod_{j=0..i} frac_i
        # weight of configs culled in iteration i
        # a_i = vol_{i-1} - vol_{i}
        #     = \prod_{j=0..i-1} frac_j - \prod_{j=0..i} frac_j
        #     = (\prod_{j=0..i-1} frac_j) (1 - frac_i)
        # log(a_i) = (\sum_{j=0..i-1} log(frac_j)) + log(1 - frac_i)
        frac_log_sums = np.append([0], np.cumsum(np.log(fracs)))
        log_a = frac_log_sums[:-1] + np.log(1.0 - fracs)

    else:

        if n_cull != 1:
            """
            # UNSUPPORTED MULTIPLE CULLS
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
            """
            raise RuntimeError(f'calc_log_a for n_cull = {n_cull} != 1 not yet implemented')

        if n_beta_samples is None or n_beta_samples == 0:
            # log_a = iters * np.log((n_walkers - n_cull + 1) / (n_walkers + 1))
            # CDW average of log vs. log of average
            log_a = iters * (-1.0 / n_walkers)
        else:
            rng = np.random.default_rng(seed=beta_seed)
            gammas = np.concatenate([np.ones((n_beta_samples, 1)),
                                     rng.beta(n_walkers, n_cull, size=(n_beta_samples, iters[-1]))],
                                     axis=1)
            # v_shell(n) = \prod_{i=1..n} gamma_i - \prod_{i=1..n-1} gamma_i
            #            = (1 - gamma_{n+1}) \prod {i=1..n} gamma_i
            # log(v_shell(n) = log(1 - gamma_{n+1}) + \sum_{i=1..n} log(gamma_i)
            log_gamma_sums = np.cumsum(np.log(gammas), axis=1)
            log_one_minus_gammas = np.log(1.0 - gammas)
            log_a = log_one_minus_gammas + log_gamma_sums

    return log_a


def calc_log_Z_terms(beta, log_a, Es, flat_V_prior=False, N_atoms=None, Vs=None):
    """Return the terms that sum to Z

    Parameters
    ----------
    beta: float
        1/(kB T)
    log_a: list(float)
        log of log NS factors
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
    log_Z_term: list of logs of terms that sum to Z, shifted by -log_shift
    log_shift: shift subtracted from each log(Z_term_true) to get log(Z_term)
    """
    log_Z_term = log_a - beta * Es
    if len(log_Z_term.shape) == 1:
        log_Z_term = log_Z_term[None, :]

    if flat_V_prior:
        if N_atoms is None or Vs is None:
            raise RuntimeError('flat_V_prior requires numbers of atoms and volumes to reweight')
        log_Z_term += N_atoms * np.log(Vs[:])

    log_shift = np.amax(log_Z_term, axis=1)
    # Z_term = np.exp(log_Z_term[:] - log_shift)
    log_Z_term = (log_Z_term.T - log_shift).T

    return (log_Z_term, log_shift)


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
    sum_f: function
        function with API equivalent to np.sum, e.g. more accurate sum

    Returns
    -------
    dict of enesmble averages of various thermodynamic quantities and extra_vals
    """
    beta = 1.0 / (kB * T)

    # Z_term here is actually Z_term_true * exp(-log_shift)
    (log_Z_term, log_shift) = calc_log_Z_terms(beta, log_a, Es, flat_V_prior, N_atoms, Vs)
    Z_term = np.exp(log_Z_term)

    # Note that
    #     Z_term = Z_term_true * exp(-log_shift)
    # exp(-log_shift) constant factor doesn't matter for quantities that are calculated from sums
    # weighted with Z_term and normalized by Z_term_sum
    Z_term_sum = sum_f(Z_term, axis=1)

    Z_term_norm = (Z_term.T / Z_term_sum).T

    U_pot = sum_f(Z_term_norm * Es, axis=1)

    if N_atoms is not None:
        N = sum_f(Z_term_norm * N_atoms, axis=1)

    U_extra_DOF = n_extra_DOF / (2.0 * beta)
    U = U_pot + U_extra_DOF + E_shift

    Cvp = n_extra_DOF * kB / 2.0 + kB * (beta ** 2) * (sum_f(Z_term_norm * Es ** 2, axis=1) - U_pot ** 2)

    if Vs is not None:
        V = sum_f(Z_term_norm * Vs, axis=1)
        thermal_exp = -1.0 / V * kB * (beta ** 2) * (sum_f(Z_term_norm * Vs, axis=1) * sum_f(Z_term_norm * Es, axis=1) -
                                                     sum_f(Z_term_norm * Vs * Es, axis=1))
    else:
        V = None
        thermal_exp = None

    if extra_vals is not None and len(extra_vals) > 0:
        extra_vals_out = []
        for v in (extra_vals):
            ev_out = np.zeros(v.shape[:-1])
            for Z_term_norm_inst in Z_term_norm:
                ev_out += sum_f(Z_term_norm_inst * v, axis=-1)
            ev_out /= Z_term_norm.shape[0]
            extra_vals_out.append(ev_out)

    # undo shift of Z_term
    log_Z = np.log(Z_term_sum) + log_shift

    # to make sure that Helmholtz_F approaches U at T -> 0,
    # we want last Z term to have w = 1, so we define a factor f which scales it correctly
    #    f exp(log_shift) Z_term[-1] = 1.0 * exp(-beta Es[-1])
    #    f = exp(-beta Es[-1] - log_shift) / Z_term[-1]
    #    log(f) = -beta Es[-1] - log_shift - log(Z_term[-1])
    log_f = -beta * Es[-1] - log_shift - log_Z_term[:, -1]
    # this factor rescales every term in Z
    log_Z += log_f

    # also add the E_shift
    Helmholtz_F = -log_Z / beta + U_extra_DOF + E_shift

    results_dict = {'log_Z': np.mean(log_Z),
                    'FG': np.mean(Helmholtz_F),
                    'U': np.mean(U),
                    'S': np.mean(U - Helmholtz_F) * beta,
                    'Cvp': np.mean(Cvp)}
    if N_atoms is not None:
        results_dict['N'] = np.mean(N)

    if Vs is not None:
        results_dict['V'] = np.mean(V)
        results_dict['thermal_exp'] = np.mean(thermal_exp)

    # compute range of configs that contributes significantly to sum
    Z_term_cumsum = np.cumsum(Z_term_norm, axis=1)

    def find_frac(term_cumsum, frac):
        term_sum = term_cumsum[-1]
        last_under = np.where(term_cumsum < frac * term_sum)[0]
        if len(last_under) == 0:
            last_under = 0
        else:
            last_under = last_under[-1]
        first_over = np.where(term_cumsum >= frac * term_sum)[0]
        if len(first_over) == 0:
            first_over = len(term_cumsum) - 1
        else:
            first_over = first_over[0]
        return np.round((first_over + last_under) / 2.0).astype(int)

    median_config = 0
    low_percentile_config = 0
    high_percentile_config = 0
    for Z_term_cumsum_inst in Z_term_cumsum:
        median_config += find_frac(Z_term_cumsum_inst, 0.5)
        low_percentile_config += find_frac(Z_term_cumsum_inst, 0.01)
        high_percentile_config += find_frac(Z_term_cumsum_inst, 0.99)
    median_config /= Z_term_cumsum.shape[0]
    low_percentile_config /= Z_term_cumsum.shape[0]
    high_percentile_config /= Z_term_cumsum.shape[0]
    median_config = int(np.round(median_config))
    low_percentile_config = int(np.round(low_percentile_config))
    high_percentile_config = int(np.round(high_percentile_config))

    p_entropy = 0.0
    for Z_term_norm_inst in Z_term_norm:
        probabilities = Z_term_norm_inst.copy()
        probabilities = probabilities[np.where(probabilities > 0.0)]
        p_entropy += -sum_f(probabilities * np.log(probabilities))
    p_entropy /= Z_term_norm.shape[0]

    results_dict.update({'low_percentile_config': low_percentile_config,
                         'median_config': median_config,
                         'high_percentile_config': high_percentile_config,
                         'p_entropy': p_entropy})

    if extra_vals is not None and len(extra_vals) > 0:
        results_dict['extra_vals'] = extra_vals_out

    # Finally, check for sampling problems
    problem = False
    # one way to get bad sampling is to be too dominated by a few configurations
    problem |= p_entropy < p_entropy_min
    # another is to clip the top (high iteration #) of the distribution, and therefore be very asymmetric
    n_avg = (high_percentile_config - low_percentile_config) // 10
    low_percentile_mean = np.mean(Z_term_norm[low_percentile_config:low_percentile_config + n_avg])
    high_percentile_mean = np.mean(Z_term_norm[high_percentile_config - n_avg:high_percentile_config])
    problem |= high_percentile_mean / low_percentile_mean > 2.0

    results_dict['problem'] = f'{problem}'.lower()

    return results_dict
