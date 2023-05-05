import numpy as np
from scipy import stats
import warnings

def calc_log_a(iters, n_walkers, n_cull, each_cull=False):
    if each_cull:
        # assume that for multiple culls, every energy is reported, use formula from
        #     SENS paper PRX v. 4 p 031034 (2014) Eq. 3
        # also assume that iters array increments by one for each cull (i.e. not exactly NS iters)
        # X_n = \prod_{i=0}^n \frac{N-i\%P}{N+1-i\%P}
        # \log X_n = \sum_{i=0}^n \log (N-i\%P) - \log(N+1-i\%P)
        i_range_mod_n_cull = np.array(range(0, iters[-1] + 1)) % n_cull
        log_X_n_term = np.log(n_walkers - i_range_mod_n_cull) - np.log(n_walkers + 1 - i_range_mod_n_cull)
        log_X_n = np.cumsum(log_X_n_term)
        # a(iter[i]) = X(iter[i-1]) - X(iter[i])
        #     = prod(0..iter[i-1]) (N-i%P)/(N+1-i%P) - prod(0..iter[i]) (N-i%P)/(N+1-i%P)
        #     = [ prod(0..iter[i-1]) (N-i%P)/(N+1-i%P) ] * (1 - prod(iter[i-1]+1..iter[i]) (N-i%P)/(N+1-i%P))
        #     = [ prod(0..iter[i-1]) (N-i%P)/(N+1-i%P) ] * (1 - prod(iter[i-1]+1..iter[i]) (N-i%P)/(N+1-i%P))
        raise RuntimeError('calc_log_a for each_cull not yet implemented')
    else:
        log_a = iters * np.log((n_walkers - n_cull + 1) / (n_walkers + 1))

    return log_a


def calc_Z_terms(beta, log_a, Es, flat_V_prior=False, N_atoms=None, Vs=None):
    log_Z_term = log_a[:] - beta*Es[:]
    if flat_V_prior:
        if N_atoms is None or Vs is None:
            raise RuntimeError('flat_V_prior requires numbers of atoms and volumes to reweight')
        log_Z_term += N_atoms*np.log(Vs[:])
    shift = np.amax(log_Z_term[:])
    Z_term = np.exp(log_Z_term[:] - shift)
    return (Z_term, shift)


def analyse_T(T, Es, E_min, Vs, extra_vals, log_a, flat_V_prior, N_atoms, kB, n_extra_DOF, KS_volume_test):
    """do an analysis at a single temperature
    T: float
        temperature
    Es: ndarray(float)
        energies at each iter
    E_min: float
        value energy was shifted by
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
    KS_volume_test: bool
        do Kolmogorov-Smirnoff Gaussian distribution test on volumes

    Returns
    -------
    dict of enesmble averages of various thermodynamic quantities and extra_vals
    """
    beta = 1.0/(kB*T)

    (Z_term, shift) = calc_Z_terms(beta, log_a, Es, flat_V_prior, N_atoms, Vs)

    Z = np.sum(Z_term)

    U_pot = np.sum(Z_term*Es) / Z

    if N_atoms is not None:
        N = np.sum(Z_term*N_atoms) / Z
        n_extra_DOF * N

    U = n_extra_DOF / (2.0 * beta) + U_pot + E_min

    Cvp = n_extra_DOF * kB / 2.0 + kB * beta * beta * (sum(Z_term * Es**2) / Z - U_pot**2)

    if Vs is not None:
        V = np.sum(Z_term*Vs)/Z
        #thermal_exp = -1.0/V * (sum(Z_term*Vs*Vs)*(-beta)*Z - np.sum(Z_term*Vs)*sum(Z_term*Vs)*(-beta)) / Z**2
        thermal_exp = -1.0/V * kB * beta*beta * (sum(Z_term*Vs)*sum(Z_term*Es)/Z - np.sum(Z_term*Vs*Es)) / Z
    else:
        V = None
        thermal_exp = None

    if extra_vals is not None and len(extra_vals) > 0:
        extra_vals_out = []
        for v in (extra_vals):
            extra_vals_out.append(np.sum(Z_term * v, axis=-1) / Z)

    log_Z = np.log(Z) + shift - beta*E_min
    # Z(T=0) contributed entirely by lowest energy, since all others are exponentially suppressed by exp(-beta E_i) term
    # Z(T=0) = a(E_min) * exp (- beta*E_min)
    # Helmholtz free energy F = - log_Z / beta
    # F(T=0) = -log_a(E_min)/beta + E_min
    # shift F so that it equals lowest internal energy (really E+PV) of all NS samples
    Helmholtz_F = -log_Z / beta + log_a[-1] / beta

    Z_max = np.amax(Z_term)
    low_percentile_config = np.where(Z_term > Z_max/10.0)[0][0]
    high_percentile_config = np.where(Z_term > Z_max/10.0)[0][-1]
    v_low_percentile_config = np.where(Z_term > Z_max/100.0)[0][0]
    v_high_percentile_config = np.where(Z_term > Z_max/100.0)[0][-1]
    mode_config = np.argmax(Z_term)

    problem = False
    if v_high_percentile_config == len(Z_term)-1:
        # warnings.warn('T may be inaccurate - significant contribution from last iteration')
        problem = True

    if KS_volume_test:
        if Vs is None:
            raise RuntimeError('KS_volume_test requires volumes')
        V_histogram = np.histogram(Vs[v_low_percentile_config:v_high_percentile_config], bins=30,
                                   weights=Z_term[v_low_percentile_config:v_high_percentile_config], density=True)
        ks_gaussianity = ks_test_gaussianity_histogram(V_histogram)
    else:
        ks_gaussianity = None

    Z_fract = np.sum(Z_term[low_percentile_config:high_percentile_config + 1]) / Z

    results_dict = {'log_Z': log_Z,
                    'FG': Helmholtz_F,
                    'U': U,
                    'S': (U - Helmholtz_F)*beta,
                    'Cvp': Cvp}

    if Vs is not None:
        results_dict['V'] = V
        results_dict['thermal_exp'] = thermal_exp

    if ks_gaussianity is not None:
        results_dict['ks_gaussianity'] = ks_gaussianity

    results_dict.update({'low_percentile_config': low_percentile_config,
                         'mode_config': mode_config,
                         'high_percentile_config': high_percentile_config,
                         'Z_fract': Z_fract})

    if extra_vals is not None and len(extra_vals) > 0:
        results_dict['extra_vals'] = extra_vals_out

    results_dict['problem'] = 'true' if problem else 'false'

    return results_dict


def ks_test_gaussianity_histogram(histogram):
    """calculate Kolmogorov-Smirnoff test for Gaussianity of histogram
    Parameters
    ----------
    histogram: histogram returned by np.histogram with density=True

    Returns
    -------
    dev_max: double max deviation from Gaussianity
    """

    dev_max = 0.0
    histo_mean =    np.sum(histogram[0] * (histogram[1][1:]-histogram[1][0:-1]) * (histogram[1][1:]+histogram[1][0:-1])/2.0 )
    histo_2nd_mom = np.sum(histogram[0] * (histogram[1][1:]-histogram[1][0:-1]) * ((histogram[1][1:]+histogram[1][0:-1])/2.0)**2 )
    histo_std_dev = np.sqrt(histo_2nd_mom - histo_mean**2)
    for ibin in range(1,len(histogram[0])):
        numerical_cumul = np.sum(histogram[0][0:ibin] * (histogram[1][1:ibin+1]-histogram[1][0:ibin]))
        analytical_cumul = stats.norm.cdf(histogram[1][ibin], histo_mean, histo_std_dev)
        dev_max = max(dev_max, np.abs( numerical_cumul - analytical_cumul))
    return dev_max

