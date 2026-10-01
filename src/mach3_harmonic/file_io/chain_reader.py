from pathlib import Path
from logging import getLogger

import uproot as ur
import numpy as np
from fnmatch import fnmatchcase
from tqdm.rich import tqdm

LARGE_LOGL = 1234567
CYCLICAL_SHIFTS = (-np.pi,np.pi)
CIRCULAR_NBINS = 200
DEFAULT_STEP_SIZE = "100 MB"

def filter_keys(keys, filters):
    """Return keys that match none of the wildcard filters."""
    return [k for k in keys if not any(fnmatchcase(k, f) for f in filters)]

def circular_mode(counts: np.ndarray, lo: float, hi: float, smooth_bins: float = 3.0) -> float:
    """1D HPD point (marginal mode) of a periodic variable from its histogram on [lo, hi),
    smoothing across the boundary. Taking counts rather than samples lets the
    histogram be filled chunk by chunk."""
    edges = np.linspace(lo, hi, len(counts) + 1)
    offsets = np.arange(-int(3 * smooth_bins), int(3 * smooth_bins) + 1)
    weights = np.exp(-0.5 * (offsets / smooth_bins) ** 2)
    smoothed = sum(w * np.roll(counts, s) for s, w in zip(offsets, weights))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return float(centres[np.argmax(smoothed)])

def apply_cyclical_shift(arr, shift,lo, hi):
    centre = 0.5*(lo+hi)
    return lo + np.mod(arr + (centre - shift) - lo, hi - lo)


# Now we have some setup
class ChainReader:
    """Reads a MaCh3 posterior TTree in chunks of `step_size` (entries or e.g. "100 MB"),
    so the file never has to fit in memory. Only the selected entries (after burn-in,
    thinning and any cut) of the kept branches are ever held in full.
    """
    def __init__(self, markov_chain: Path, posterior_tree: str,
                logl_branch: str = "logL", pars_to_ignore: list[str] | None = None,
                cyclical_pars: list[str]|None=None, burn_in: int=0, thin: int=1,
                step_size: int | str = DEFAULT_STEP_SIZE):

        getLogger().info(f"Opening {posterior_tree} in {markov_chain} with burn_in={burn_in}, thin={thin}, step_size={step_size}")

        if not markov_chain.is_file():
            raise FileNotFoundError(f"Cannot find MCMC {markov_chain}")

        try:
            self._chain = ur.open(f"{markov_chain}:{posterior_tree}")
        except ur.KeyInFileError:
            raise ValueError(f"Cannot find {posterior_tree} in {markov_chain}") from None

        if not isinstance(self._chain, ur.TTree):
            raise TypeError(f"{posterior_tree} in {markov_chain} is not a TTree")

        if logl_branch not in self._chain.keys():
            raise ValueError(f"Cannot find {logl_branch} in posteriors")

        # uproot does the wildcard matching; we just subtract the matches.
        # logL always goes last so param_names/ndim can drop it.
        ignored = set(self._chain.keys(filter_name=[*(pars_to_ignore or [])]))
        self._branches_to_keep = [k for k in self._chain.keys() if k not in ignored and k != logl_branch]
        self._branches_to_keep.append(logl_branch)

        if not 0 <= burn_in < self._chain.num_entries:
            raise ValueError(f"burn_in ({burn_in}) must be in [0, {self._chain.num_entries})")
        if thin < 1:
            raise ValueError(f"thin must be >= 1, got {thin}")

        self._logl_branch = logl_branch
        self.burn_in = burn_in
        self.thin = thin
        self.step_size = step_size

        # Uncut, so every load uses the same shift
        self.cyclical_shifts = self._find_cyclical_shifts(
            [par for par in cyclical_pars or [] if par in self._branches_to_keep]
        )

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

    # ----------------------------------------------------------------------- #
    # Streaming
    # ----------------------------------------------------------------------- #

    @property
    def _n_after_burn_in(self) -> int:
        return self._chain.num_entries - self.burn_in

    def _iterate(self, branches: list[str] | None = None, expressions: list[str] | None = None,
                 desc: str = "Reading chain"):
        """Yield (arrays, slice over post-burn-in entries) one chunk at a time."""
        kwargs = {}
        if branches is not None:
            wanted = set(branches)
            kwargs["filter_name"] = lambda name: name in wanted  # exact match, no globbing

        with tqdm(total=self._n_after_burn_in, desc=desc, unit=" steps") as progress:
            for arrays, report in self._chain.iterate(
                expressions, library="np", step_size=self.step_size,
                entry_start=self.burn_in, report=True, **kwargs,
            ):
                yield arrays, slice(report.tree_entry_start - self.burn_in,
                                    report.tree_entry_stop - self.burn_in)
                progress.update(report.tree_entry_stop - report.tree_entry_start)

    def _entry_mask(self, cut: str | None) -> np.ndarray | None:
        """Boolean mask over post-burn-in entries for thinning and the cut.

        None if every entry is kept. The cut is evaluated in its own pass so the
        output can be allocated at its exact size; the mask is 1 byte per entry.
        """
        if cut is None and self.thin == 1:
            return None

        mask = np.zeros(self._n_after_burn_in, dtype=bool)
        mask[::self.thin] = True
        if cut is not None:
            for arrays, entries in self._iterate(expressions=[cut], desc=f"Applying cut {cut}"):
                mask[entries] &= next(iter(arrays.values())).astype(bool)
        return mask

    def _stream(self, branches: list[str], mask: np.ndarray | None, desc: str = "Loading chain"):
        """Yield (selected arrays, slice into the output) one chunk at a time."""
        pos = 0
        for arrays, entries in self._iterate(branches, desc=desc):
            if mask is not None:
                chunk_mask = mask[entries]
                arrays = {k: v[chunk_mask] for k, v in arrays.items()}
            n = len(arrays[branches[0]])
            yield arrays, slice(pos, pos + n)
            pos += n

    def _n_selected(self, mask: np.ndarray | None) -> int:
        return self._n_after_burn_in if mask is None else int(mask.sum())

    def _find_cyclical_shifts(self, pars: list[str]) -> dict[str, float]:
        if not pars:
            return {}
        counts = {par: np.zeros(CIRCULAR_NBINS) for par in pars}
        for arrays, _ in self._iterate(pars, desc="Finding cyclical shifts"):
            for par in pars:
                counts[par] += np.histogram(arrays[par], bins=CIRCULAR_NBINS, range=CYCLICAL_SHIFTS)[0]
        return {par: circular_mode(c, *CYCLICAL_SHIFTS) for par, c in counts.items()}

    # ----------------------------------------------------------------------- #
    # Loading
    # ----------------------------------------------------------------------- #

    def get_chain(self, cut: str|None=None):
        mask = self._entry_mask(cut)
        n = self._n_selected(mask)

        # Filled in place: these are the only full-length arrays we hold
        samples = np.empty((1, n, self.ndim))
        lnprob = np.empty((1, n))

        for arrays, out in self._stream(self._branches_to_keep, mask):
            for idx, name in enumerate(self.param_names):
                col = arrays[name]
                if name in self.cyclical_shifts:
                    col = apply_cyclical_shift(col, self.cyclical_shifts[name], *CYCLICAL_SHIFTS)
                samples[0, out, idx] = col
            lnprob[0, out] = -arrays[self._logl_branch]

        lnprob[lnprob < -LARGE_LOGL] = -np.inf
        return samples, lnprob

    def get_single_branch(self, branch_name: str, cut: str|None=None):
        if branch_name not in self._chain.keys():
            raise ValueError(f"Cannot find {branch_name} in posteriors")

        mask = self._entry_mask(cut)
        branch = np.empty(self._n_selected(mask))
        for arrays, out in self._stream([branch_name], mask, desc=f"Loading {branch_name}"):
            branch[out] = arrays[branch_name]
        return branch


    def undo_cyclical_shifts(self, chain):
        for par, shift in self.cyclical_shifts.items():
            par_idx = self._branches_to_keep.index(par)
            chain[:,par_idx] = apply_cyclical_shift(chain[:,par_idx], -shift, *CYCLICAL_SHIFTS)

