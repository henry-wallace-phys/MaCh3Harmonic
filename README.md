# MaCh3Harmonic

Bayesian evidence and Bayes factors for [MaCh3](https://github.com/mach3-software/MaCh3) MCMC chains, using the learnt harmonic mean estimator from [harmonic](https://github.com/astro-informatics/harmonic).

MaCh3Harmonic reads a MaCh3 posterior TTree and splits it into regions with a cut (for example the two mass orderings). For each region it trains a normalising flow on half of the samples and uses the other half to estimate the evidence. The ratio of the two evidences is the Bayes factor. It is reported next to the simple MCMC step-count ratio as a cross-check.

ROOT files are streamed in chunks, so chains much larger than RAM can be read. Only the samples you select (after burn-in, `max_entries`, thinning and the cut) are held in memory.

## Installation

Requires Python ≥ 3.11.

```bash
git clone git@github.com:henry-wallace-phys/MaCh3Harmonic.git
cd MaCh3Harmonic
pip install -e .
```

*NOTE*: Right now I'm using a slightly modified harmonic install so I can pass in early stopping

For GPU training, install JAX with CUDA support as described in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Usage

```bash
mach3harmonic -c config.yaml single_flow_comp
```

| Option | Description |
| --- | --- |
| `-c`, `--config` | YAML config file (required) |
| `-l`, `--log-level` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR` or `CRITICAL` |

### `single_flow_comp`

Splits one chain into two regions on a single parameter and computes the Bayes factor between them. With `param_to_cut: delm2_23`, `cut_value: 0` and `labels: ["IO", "NO"]`:

1. Loads the steps with `delm2_23 < 0`. A flow is trained on the first half of them and the evidence Z_IO is computed from the second half.
2. Does the same for `delm2_23 > 0` to get Z_NO.
3. Logs the harmonic Bayes factor Z_NO / Z_IO with its error.
4. Logs the MCMC Bayes factor N(`delm2_23 > 0`) / N(`delm2_23 < 0`). Its error accounts for autocorrelation in the chain.

`labels` names the two regions, below the cut first. Each Bayes factor is reported as second label / first label.

Files written to the working directory for each label:

| File | Contents |
| --- | --- |
| `<label>.flow`, `<label>.flow.json` | Trained flow and a JSON file describing it (versions, training settings, losses, and a hash of the training samples) |
| `triangle_<label>.pdf` | Triangle plot of the flow against the inference samples |
| `1d_<label>.pdf` | 1D marginals of the flow against the inference samples, one page per parameter |
| `loss_<label>_loss.pdf` | Training loss curve |

## Configuration

See [docs/example_config.yaml](docs/example_config.yaml) for a complete example.

### `Chains`

`global` holds defaults for every chain. Each entry under `files` is a chain, referred to by its key, and can override any global setting. List settings (`pars_to_ignore`, `cyclical_pars`) are added to the global list rather than replacing it.

```yaml
Chains:
    global:
        pars_to_ignore: [logL*, step*]
        cyclical_pars: [delta_cp]
        burn_in: 10000
    files:
        my_chain:
          chain_path: my_chain.root
          posterior_tree: posteriors
          pars_to_ignore: [prod_height]   # ignored in addition to logL*, step*
```

| Key | Default | Description |
| --- | --- | --- |
| `chain_path` | — | Path to the ROOT file. Set this per chain. |
| `posterior_tree` | `posteriors` | Name of the posterior TTree. Set this per chain. |
| `pars_to_ignore` | `[]` | Branches to leave out of the fit. Wildcards are allowed (e.g. `step*`). |
| `logl_branch` | `logL` | Branch holding −log L. It is never treated as a parameter, even if it isn't in `pars_to_ignore`. |
| `cyclical_pars` | `[]` | Angles on [−π, π) to re-centre on their mode before training (see below) |
| `burn_in` | `0` | Number of steps to drop from the start of the chain |
| `max_entries` | all | Only read this many steps after burn-in |
| `thin` | `1` | Keep every n-th step after burn-in |
| `step_size` | `100 MB` | How much of the file to read at a time, as a size (`"500 MB"`) or a number of steps |

Steps are selected in this order: burn-in, the `max_entries` window, thinning, then the cut. Every pass over the file shows a progress bar.

`cyclical_pars`: a flow can't model a distribution that wraps around ±π. Each listed parameter is rotated so the peak of its post-burn-in distribution sits at the centre of [−π, π), which moves the wrap-around point to the region with the fewest samples. The same rotation is used for every cut.

**Large or `hadd`ed chains.** `max_entries` takes steps from the start of the file. If the file is many chains joined with `hadd`, that is only the first few chains, which may never visit one of the cut regions. A cut that selects no steps stops with an error. To get a smaller sample spread over the whole file, use `thin` instead.

### `Flows`

The flow is a rational-quadratic spline flow (5 hidden layers of 64 units, 20 bins). Each region's samples are split into 100 consecutive blocks: the first 50 are used for training and the last 50 for the evidence. Steps beyond a multiple of 100 are dropped.

| Key | Default | Description |
| --- | --- | --- |
| `training.epochs_num` | `50` | Maximum number of training epochs |
| `training.early_stopping` | `20` | Stop after this many epochs without improvement |
| `training.learning_rate` | `1e-4` | Optimiser learning rate |
| `training.batch_size` | `4096` | Training batch size |
| `training.temperature` | `1.0` | Flow temperature for evidence evaluation and sampling. Values below 1 concentrate the flow and keep its tails inside the posterior. |
| `file_io.flow_path` | — | Also save the trained flow here. `single_flow_comp` overwrites it for each region, so use the `<label>.flow` files instead. |

### `Evidence`

| Key | Default | Description |
| --- | --- | --- |
| `predict_batch_size` | `10000` | Number of samples sent to the flow at once when computing the evidence. Lower this if the GPU runs out of memory. It does not change the result. |

### `SingleFlowComp`

| Key | Description |
| --- | --- |
| `chain_label` | Which chain under `Chains.files` to use |
| `param_to_cut` | Parameter that defines the two regions |
| `cut_value` | Value that separates them |
| `labels` | Names of the regions, below the cut first. Defaults to the cut expressions. |

The example config also has `Flows.ensemble`, `Evidence.seed`, `Evidence.n_samples` and `Plotting` sections. None of these are used yet.
