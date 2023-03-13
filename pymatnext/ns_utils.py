import warnings

import numpy as np

def rngs(i_local, random_seed=None, rng_global=None):
    """generate single global and per-process local rngs
    
    Parameters
    ----------
    i_local: int
        index of local process, to generate correct local rng
    random_seed: int, default None
        seed for rng - everything is deterministic, including local rngs, for each value,
        or use default entropy source if None or <= 0
    rng_global: np.random.Generator, default None
        if present, rng_global to start with, instead of generating one from seed:w

    Returns
    -------
    rng_global, rng_local: numpy.random.Generator objects
    """
    if random_seed is not None and random_seed <= 0:
        random_seed = None

    if rng_global is None:
        # need a new rng_global, and seed will be used for local generators as well
        rng_global = np.random.default_rng(seed = random_seed)
    else:
        # get seed from exiting global generator
        random_seed = rng_global.integers(np.iinfo(np.int64).max)
        warnings.warn(f"Initializing random seed from rng_global to {random_seed}")

    # see https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html#numpy.random.PCG64
    local_seeds = np.random.SeedSequence(random_seed + 10 if random_seed is not None else None)
    rng_local = [np.random.default_rng(seed=s) for s in local_seeds.spawn(i_local+1)]
    rng_local = rng_local[-1]

    # use rng.bit_generator.state to save/restore state
    return rng_global, rng_local
