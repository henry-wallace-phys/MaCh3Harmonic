# Helper functions to set up objects using the YAML
from pathlib import Path
from dataclasses import dataclass

import harmonic as hm

from mach3_harmonic.file_io import ChainReader
from mach3_harmonic.stats import train_model, mach3_to_chain, get_evidence

_DEFAULT_GLOBAL_CHAIN = {
    "pars_to_ignore": [],
    "cyclical_pars": [],
    "logl_branch": "logL",
    "burn_in": 0,
    "thin": 1,
    "step_size": "100 MB",  # chunk size when streaming the TTree (entries or bytes)
}

_DEFAULT_TRAINING = {
    "epochs_num": 50,
    "temperature": 1.0,
    "early_stopping": 20,
    "learning_rate": 1e-4,
    "batch_size": 4096,
}


def _merge_settings(base: dict, extra: dict, allowed: dict) -> dict:
    """Merge keys from `extra` that exist in `allowed` into a copy of `base`.

    Lists are extended (skipping duplicates); all other values are overridden.
    Keys in `extra` that aren't in `allowed` are ignored.
    """
    merged = {k: (v.copy() if isinstance(v, list) else v) for k, v in base.items()}
    for key, value in extra.items():
        if key not in allowed:
            continue
        if isinstance(merged.get(key), list) and isinstance(value, list):
            merged[key] += [v for v in value if v not in merged[key]]
        else:
            merged[key] = value
    return merged


def load_chain(yaml_config: dict, chain_labels: list[str] | str | None = None) -> dict[str, ChainReader]:
    if isinstance(chain_labels, str):
        chain_labels = [chain_labels]

    chain_settings = yaml_config.get("Chains")

    if chain_settings is None:
        raise ValueError("Cannot find 'Chains' in input YAML")

    chain_files = chain_settings.get("files")
    if chain_files is None:
        raise ValueError("No 'files' listed in 'Chains' settings")

    global_settings = _merge_settings(
        _DEFAULT_GLOBAL_CHAIN,
        chain_settings.get("global") or {},
        _DEFAULT_GLOBAL_CHAIN,
    )

    if chain_labels is None:
        chain_labels = list(chain_files.keys())

    missing = [label for label in chain_labels if label not in chain_files]
    if missing:
        raise KeyError(f"Chains not found in 'Chains.files': {missing}")

    chains = {}
    for label in chain_labels:
        chain_info = chain_files[label] or {}

        chain_path = chain_info.get("chain_path")
        if chain_path is None:
            raise ValueError(f"No 'chain_path' provided for chain '{label}'")

        chain_posterior = chain_info.get("posterior_tree", "posteriors")

        chain_additional_settings = _merge_settings(
            global_settings, chain_info, _DEFAULT_GLOBAL_CHAIN
        )

        chains[label] = ChainReader(
            Path(chain_path),
            chain_posterior,
            **chain_additional_settings,  # adjust to match ChainReader's signature
        )

    return chains

def train_flow(yaml_config, chain: hm.Chains, ndim: int, override_loss_plot: str|None=None):
    flow_settings = yaml_config.get("Flows")

    training_settings = _merge_settings(_DEFAULT_TRAINING, flow_settings.get('training', {}), _DEFAULT_TRAINING)
    
    file_io_settings = flow_settings.get('file_io', {})
    training_settings["flow_path"] = file_io_settings.get("flow_path")
    training_settings["loss_path"] = override_loss_plot or file_io_settings.get("loss_path")
    
    return train_model(chain, ndim, **training_settings)

def run_inference(yaml_config, chain: ChainReader, cut: str|None, override_loss_plot: str|None=None):
    samples, lnprob = chain.get_chain(cut)
    
    train_chain, infer_chain = mach3_to_chain(samples, lnprob, chain.ndim)
    del samples, lnprob  # harmonic holds its own copies; don't keep ours through training
    
    model = train_flow(yaml_config, train_chain, ndim = chain.ndim, override_loss_plot=override_loss_plot)
    
    # Now we get the evidence 
    evidence = get_evidence(model, infer_chain)
    evidence.add_chains(infer_chain)
    
    return train_chain, infer_chain, model, evidence
