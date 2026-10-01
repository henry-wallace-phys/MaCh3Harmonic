import hashlib
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from logging import getLogger

import numpy as np
import harmonic as hm
import jax
import jax.numpy as jnp
from tqdm.rich import tqdm
from matplotlib import pyplot as plt


PREDICT_BATCH = 10_000  # flow intermediates scale with batch * ndim * n_bins; lower if the GPU OOMs
FLOW_FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #

def mach3_to_chain(samples, lnprob, ndim, nblocks: int = 100,
                   training_proportion: float = 0.5) -> tuple[hm.Chains, hm.Chains]:
    """MaCh3 (nchains, nsteps, ndim) arrays -> (chains_train, chains_infer).

    harmonic's split is deterministic (first blocks train, the rest infer), so the
    same inputs always give the same split; save/load relies on that.
    """
    if samples.shape[1] < nblocks:
        raise ValueError(f"Only {samples.shape[1]} samples; need at least nblocks={nblocks} "
                         "to split into training/inference blocks.")
    n_keep = samples.shape[1] // nblocks * nblocks
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples[:, :n_keep], lnprob[:, :n_keep])
    chains.split_into_blocks(nblocks)

    return hm.utils.split_data(chains, training_proportion=training_proportion)


# --------------------------------------------------------------------------- #
# Saving / loading
# --------------------------------------------------------------------------- #

def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _fingerprint(chains_train: hm.Chains) -> str:
    """Hash of the training samples, so a flow can't be silently reused on a
    different chain (its evidence would then be evaluated on data it may have
    been trained on)."""
    x = np.ascontiguousarray(np.asarray(chains_train.samples))
    h = hashlib.sha1()
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def save_flow(model, path, chains_train: hm.Chains | None = None, **metadata) -> Path:
    """Save a trained harmonic flow plus a JSON sidecar of metadata.

    The model goes through harmonic's own serialize() (cloudpickle), so reload it
    with the same harmonic/jax/flax versions. The sidecar is plain JSON so you can
    see what a file holds without unpickling it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Don't pickle the chunked-predict wrapper from _batch_predict: it closes over
    # tqdm and the original bound method, and would be wrapped again on reload.
    patched = model.__dict__.pop("predict", None)
    try:
        model.serialize(str(path))
    finally:
        if patched is not None:
            model.predict = patched

    meta = {
        "format_version": FLOW_FORMAT_VERSION,
        "model_class": type(model).__name__,
        "ndim": int(model.ndim),
        "temperature": float(model.temperature),
        "harmonic_version": _pkg_version("harmonic"),
        "jax_version": jax.__version__,
        "fingerprint": _fingerprint(chains_train) if chains_train is not None else None,
        **metadata,
    }
    _meta_path(path).write_text(json.dumps(meta, indent=2, default=str))
    getLogger().info(f"Saved flow to {path}")
    return path


def load_flow(path, chains_train: hm.Chains | None = None):
    """Load a flow saved with save_flow. Returns (model, metadata).

    If chains_train is given, checks it's the chain the flow was trained on.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    meta_file = _meta_path(path)
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    if chains_train is not None:
        if meta.get("ndim", chains_train.ndim) != chains_train.ndim:
            raise ValueError(f"{path} is a {meta['ndim']}D flow, "
                             f"chain is {chains_train.ndim}D")
        saved = meta.get("fingerprint")
        if saved is not None and saved != _fingerprint(chains_train):
            raise ValueError(
                f"{path} was trained on different samples than the ones supplied. "
                "Retrain (retrain=True) or pass the original chain."
            )

    model = hm.model.FlowModel.deserialize(str(path))
    getLogger().info(f"Loaded {meta.get('model_class', 'flow')} from {path}")
    return model, meta


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def plot_losses(losses: np.ndarray, plot_name: str):
    fig, ax = plt.subplots(1)
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(f"loss_{plot_name}")
    plt.close(fig)


def train_model(chains_train: hm.Chains, ndim: int, epochs_num: int=20,
                temperature: float = 0.8, early_stopping: int = 20,
                learning_rate: float = 1e-4, batch_size: int = 4096,
                flow_path: str | Path | None = None, retrain: bool = False, loss_path: str|None=None):
    """Train an RQ-spline flow, or load it from flow_path if already trained.

    Returns (model, losses). Losses are stored in the sidecar, so a loaded flow
    returns the same loss curve it was trained with.

    `temperature` is applied to a loaded flow too: harmonic only reads it in
    predict/sample, so temperatures can be scanned without retraining.
    """
    flow_path = Path(flow_path) if flow_path is not None else None

    model = hm.model.RQSplineModel(
        ndim, learning_rate=learning_rate, standardize=True, n_bins=20,
        hidden_size=[64, 64, 64, 64, 64], spline_range=(-10, 10),
        temperature=temperature,
    )
    losses = np.asarray(model.fit(chains_train.samples, epochs=epochs_num, verbose=True,
                                  early_stopping=early_stopping, batch_size=batch_size))

    if flow_path is not None:
        save_flow(
            model, flow_path, chains_train=chains_train,
            epochs_trained=len(losses),
            losses=losses.tolist(),
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping=early_stopping,
        )
    
    if loss_path is not None:
        plot_losses(losses, loss_path)

    return model


# --------------------------------------------------------------------------- #
# Plotting / evidence
# --------------------------------------------------------------------------- #

def _batch_predict(model, batch_size: int = PREDICT_BATCH):
    """Make model.predict evaluate in chunks so large inference sets fit in GPU memory.

    harmonic's Evidence calls model.predict(x=X) on every inference sample at once;
    shadowing the method on the instance keeps the harmonic code untouched.
    """
    if "predict" in model.__dict__:  # already wrapped
        return model
    predict_full = model.predict

    def predict(x, *args, **kwargs):
        return jnp.concatenate([
            predict_full(x[i:i + batch_size], *args, **kwargs)
            for i in tqdm(range(0, len(x), batch_size), desc="sampling batches")
        ])

    model.predict = predict
    return model


def batched_sample(model, n: int, rng_key, batch_size: int = PREDICT_BATCH) -> np.ndarray:
    """Draw n flow samples in chunks so large draws fit in GPU memory.

    Each chunk gets its own key, otherwise every chunk would repeat the same draws.
    """
    keys = jax.random.split(rng_key, max(1, -(-n // batch_size)))
    return np.concatenate([
        np.asarray(model.sample(min(batch_size, n - i), rng_key=key))
        for i, key in zip(range(0, n, batch_size), keys)
    ]) if n > 0 else np.empty((0, model.ndim))


def get_evidence(model, chains_infer: hm.Chains, shift: float = hm.Evidence.Shifting.ABS_MAX,
                 predict_batch_size: int = PREDICT_BATCH) -> hm.Evidence:
    """Evidence from a trained (or loaded) flow on the held-out chains."""
    _batch_predict(model, predict_batch_size)
    ev = hm.Evidence(chains_infer.nchains, model, shift=shift)
    ev.add_chains(chains_infer)

    if np.isnan(ev.compute_ln_evidence()).any():
        raise ValueError("Evidence is NaN! Cannot compute Bayes factor...")
    return ev


def sample_evidence_weighted_flows(
    model_no, model_io,
    ln_z_no: float, ln_z_io: float,
    n_samples: int,
    ordering_idx: int | None = None,
    oversample: float = 1.5,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Draw from p(x) = w_NO q_NO(x) + w_IO q_IO(x), with w ∝ evidence.

    If ordering_idx is given, flow samples that leak across delm2_23 = 0
    (the flow's support is unbounded, the true subset posterior is not)
    are rejected so each component stays in its own ordering.

    Either model can be a path to a flow saved with save_flow.
    """
    if isinstance(model_no, (str, Path)):
        model_no, _ = load_flow(model_no)
    if isinstance(model_io, (str, Path)):
        model_io, _ = load_flow(model_io)

    rng = np.random.default_rng(seed)
    w_no = 1.0 / (1.0 + np.exp(ln_z_io - ln_z_no))  # stable in log space
    n_no = rng.binomial(n_samples, w_no)
    n_io = n_samples - n_no

    # harmonic's sample() defaults to PRNGKey(0), so without explicit keys both
    # flows reuse the same noise and the draws never change with `seed`.
    key_no, key_io = jax.random.split(jax.random.PRNGKey(rng.integers(2**31)))

    def draw(model, n, sign, name, key):
        if n == 0:
            return None
        s = batched_sample(model, int(np.ceil(oversample * n)), key)
        if ordering_idx is not None:
            keep = np.sign(s[:, ordering_idx]) == sign
            getLogger().warning(f"{name} flow: {1 - keep.mean():.2%} of samples leaked across the ordering boundary")
            s = s[keep]
        if len(s) < n:
            getLogger().warning(f"WARNING: only {len(s)}/{n} valid {name} samples; "
                  "increase `oversample`. Mixture weights will be slightly off.")
        return s[:n]

    parts = [p for p in (draw(model_no, n_no, +1, "NO", key_no),
                         draw(model_io, n_io, -1, "IO", key_io)) if p is not None]
    mix = np.concatenate(parts)
    return mix[rng.permutation(len(mix))], w_no