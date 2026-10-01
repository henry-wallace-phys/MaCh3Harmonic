import numpy as np
from logging import getLogger

from mach3_harmonic.file_io import ChainReader

def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of a 1D series.

    ACF computed via FFT, summed up to Sokal's automatic window
    (smallest M such that M >= c * tau(M)), as in emcee.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if n < 2 or x.var() == 0:
        return 1.0

    nfft = 1 << (2 * n - 1).bit_length()  # zero-pad to avoid circular correlation
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    acf /= acf[0]

    taus = 2.0 * np.cumsum(acf) - 1.0
    within = np.arange(n) < c * taus
    window = np.argmin(within) if not within.all() else n - 1
    return max(float(taus[window]), 1.0)


def mcmc_bayes_factor(chain_reader: ChainReader, bayes_factor_var: str, cut_val: float, burn_in: int=0):
    '''We're doing a simple var>cut_val'''
    chain = chain_reader.get_single_branch(bayes_factor_var, burn_in=burn_in)
    
    n = len(chain)
    
    above_cut = (chain>cut_val).astype(float)
    n_above_cut = int(above_cut.sum())
    
    n_below_cut = n - n_above_cut
    
    if n_above_cut == 0 or n_below_cut == 0:
        raise ValueError(f"Chain never visits one side of the cut (N abve={n_above_cut}, N below={n_below_cut}); "
                         "cannot estimate the Bayes factor from step counts.")
    
    p = n_above_cut/n
    bf = n_above_cut/n_below_cut
    
    tau = integrated_autocorr_time(above_cut)
    
    n_eff = n / tau

    sigma_p = np.sqrt(p * (1 - p) / n_eff)
    sigma_bf = sigma_p / (1 - p) ** 2
    sigma_bf_iid = sigma_p * np.sqrt(tau) / (1 - p) ** 2  # ignoring correlations

    if n < 50 * tau:
        getLogger().warning(f"WARNING: chain length ({n}) < 50 x tau ({tau:.1f}); the ordering "
              "flips too rarely for a reliable autocorrelation estimate, so the "
              "MCMC Bayes factor error is likely underestimated.")

    return {
        "bayes_factor": bf,
        "error": sigma_bf,
        "error_iid": sigma_bf_iid,
        "tau": tau,
        "n_eff": n_eff,
        "n_no": n_above_cut,
        "n_io": n_below_cut,
    }
