from pathlib import Path
from logging import getLogger

import uproot as ur
import numpy as np
from fnmatch import fnmatchcase

LARGE_LOGL = 1234567
CYCLICAL_SHIFTS = (-np.pi,np.pi)

def filter_keys(keys, filters):
    """Return keys that match none of the wildcard filters."""
    return [k for k in keys if not any(fnmatchcase(k, f) for f in filters)]

def circular_mode(x: np.ndarray, lo: float, hi: float, nbins: int = 200, smooth_bins: float = 3.0) -> float:
    """1D HPD point (marginal mode) of a periodic variable, smoothing across the boundary."""
    counts, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    offsets = np.arange(-int(3 * smooth_bins), int(3 * smooth_bins) + 1)
    weights = np.exp(-0.5 * (offsets / smooth_bins) ** 2)
    smoothed = sum(w * np.roll(counts, s) for s, w in zip(offsets, weights))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return float(centres[np.argmax(smoothed)])
    
def apply_cyclical_shift(arr, shift,lo, hi):
    centre = 0.5*(lo+hi)
    return lo + np.mod(arr + (centre - shift) - lo, hi - lo)
    

# Now we have some setup
class ChainReader:
    def __init__(self, markov_chain: Path, posterior_tree: str,
                logl_branch: str = "logL", pars_to_ignore: list[str] | None = None,
                cyclical_pars: list[str]|None=None, burn_in: int=0):

        if not markov_chain.is_file():
            raise FileNotFoundError(f"Cannot find MCMC {markov_chain}")

        try:
            self._chain = ur.open(f"{markov_chain}:{posterior_tree}")
        except ur.KeyInFileError:
            raise ValueError(f"Cannot find {posterior_tree} in {markov_chain}") from None

        if not isinstance(self._chain, ur.TTree):
            raise TypeError(f"{posterior_tree} in {markov_chain} is not a TTree")

        # uproot does the wildcard matching; we just subtract the matches
        ignored = set(self._chain.keys(filter_name=[*(pars_to_ignore or [])]))
        self._branches_to_keep = [k for k in self._chain.keys() if k not in ignored]
        
        
        if logl_branch not in self._branches_to_keep:
            if logl_branch not in self._chain.keys():
                raise ValueError(f"Cannot find {logl_branch} in posteriors")
            
            self._branches_to_keep.append(logl_branch)

        self._logl_branch = logl_branch
        self.burn_in = burn_in

        self.cyclical_shifts = {}
        for par in cyclical_pars or []:
            if par not in self._branches_to_keep:
                continue
            full = self._chain[par].array(library="np")  # uncut, so every load uses the same shift
            self.cyclical_shifts[par] = circular_mode(full, CYCLICAL_SHIFTS[0], CYCLICAL_SHIFTS[1])

        getLogger().info(f"Opened {posterior_tree} in {markov_chain}")
    
    @property
    def ndim(self):
        return len(self._branches_to_keep)-1
    
    @property
    def param_names(self):
        return self._branches_to_keep[:-1]
    
    def get_par_idx(self, par_name: str):
        if par_name not in self._branches_to_keep:
            raise ValueError(f"Cannot find {par_name} in stored ttree")
        
        return self._branches_to_keep.index(par_name)
    
    def get_chain(self, cut: str|None=None):
        chain_arr = self._chain.arrays(library="np", filter_name=self._branches_to_keep, cut=cut)
        
        
        for cyc, shift in self.cyclical_shifts.items():
            chain_arr[cyc] = apply_cyclical_shift(chain_arr[cyc], shift, *CYCLICAL_SHIFTS)
        
        samples = np.column_stack([chain_arr[name] for name in self._branches_to_keep if name!=self._logl_branch])[self.burn_in:]
        lnprob = -chain_arr[self._logl_branch][self.burn_in:]
        lnprob[lnprob< -LARGE_LOGL] = -np.inf
        
        return np.ascontiguousarray(samples[None, ...]), np.ascontiguousarray(lnprob[None, ...])
    
    def get_single_branch(self, branch_name: str, cut: str|None=None, burn_in: int=0):
        chain_arr = self._chain.arrays(library="np", filter_name=branch_name, cut=cut)
        return chain_arr[branch_name]

    
    def undo_cyclical_shifts(self, chain):
        for shift, par in self.cyclical_shifts:
            par_idx = self._branches_to_keep.index(par)
            chain[:,par_idx] = apply_cyclical_shift(chain[:,par_idx], -shift, *CYCLICAL_SHIFTS)
    
