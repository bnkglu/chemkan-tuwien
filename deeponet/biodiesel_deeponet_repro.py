r"""Source-faithful preprocessing and metrics for the biodiesel DeepONet REPRODUCTION path.

The model itself (308 parameters, ReLU, Glorot normal, zero biases) is unchanged and lives
in ``biodiesel_deeponet.py``. What this module replaces, for new runs only, is the legacy
preprocessing of ``biodiesel_deeponet.prepare_inputs`` (trunk tau = t / t_end, train-only
min-max), which stays for loading legacy checkpoints.

SOURCES
    * Time grid, initial conditions, split, noise: the ChemKAN authors' released
      ``ChemKAN_biodiesel_example.jl`` as reproduced by ``biodiesel_v2.npz``:
      t = range(0, 30, length=30); noise = clean + randn * p * max_t(clean) per trajectory
      and species, clipped at 0 (``data_gen/common.add_noise``, additive_species_max).
    * Trunk input: the raw time coordinate t, as lululxvi/deeponet / DeepXDE 0.11.2 feed
      the coordinate itself to the trunk.
    * State normalization: their DeepONet code was NOT released, so whether it saw raw or
      normalized states is unknown. Two diagnostic modes (``STATE_MODES``):

        raw_states
            branch [TG0, ROH0, T0] in physical units; targets = raw six-species states.
        author_global_normalized_states
            the normalization of ChemKAN_biodiesel_example.jl:
                ymax = maximum(maximum(ode_data_list, dims=1), dims=3)
                ymin = minimum(minimum(ode_data_list, dims=1), dims=3)
            one min/max per species over ALL experiments and times of ode_data_list,
            which holds the 20 (noisy) training, 10 (noisy) testing and 10 noise-free
            testing trajectories; TG0/ROH0 use their species' scale; T0 uses the min/max
            of all 30 initial temperatures (the duplicated test copy adds no new values).

METRICS (tensors [trajectory, time, species]; the time length is taken from the tensor)
    mean_mse   (pred - target)^2 averaged over everything, in the TRAINED representation.
    eq18_obs   Eq. 18 in normalized u units against the observations:
               per trajectory mean over species, SUM over time; mean over trajectories.
    eq22_true  the same reduction against the noise-free truth (Eq. 22).
Eq. 18/22 are always evaluated in author-global normalized units, whatever the state mode,
so raw-state runs are scored in the units the paper's losses use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = _REPO / "chemkan" / "data" / "generated" / "biodiesel_v2.npz"
sys.path.insert(0, str(_REPO / "chemkan" / "scripts"))

from _data import noise_tag   # noqa: E402

STATE_MODES = ("raw_states", "author_global_normalized_states")
OBJECTIVES = ("mean_mse", "eq18_obs", "eq22_true")
TRUNK_INPUT = "raw_t"      # the ONLY supported trunk input: raw physical time (s)
BRANCH_SPECIES = (0, 1)       # TG, ROH columns of the six-species state
MILESTONES = (0, 500, 1000, 2500, 5000, 7500)   # + the final epoch count
MILESTONE_KEYS = ("train_mean_mse", "test_mean_mse", "train_eq18_obs", "test_eq18_obs")
FIG3_TEST_INDEX = 3           # test row 24 of diesel_u0_samples.txt (paper Fig. 3 condition)
# Axes of every state tensor here: [trajectory, time, species].
TIME_DIM, SPECIES_DIM = 1, 2


def load_author_data(data_file: Path | str = DEFAULT_DATA, noise_percent: int = 0) -> dict:
    """Clean and observed states of both splits, as float64 numpy, [traj, time, species].

    ``train_obs``/``test_obs`` are the observations at ``noise_percent`` (the clean states at
    0 %); ``train_true``/``test_true`` are noise-free. Branch initial conditions come from
    the clean t = 0 states, as ``u0_list`` does in the Julia script.
    """
    d = np.load(data_file, allow_pickle=True)
    t = np.asarray(d["t"], dtype=np.float64)
    tag = noise_tag(noise_percent)
    out = {"t": t, "species": [str(s) for s in d["species"]], "noise_percent": int(noise_percent),
           "data_file": str(data_file)}
    for split in ("train", "test"):
        true = np.asarray(d[f"{split}_states"], dtype=np.float64)
        obs = true if noise_percent == 0 else np.asarray(d[f"{split}_states_{tag}"], dtype=np.float64)
        out[f"{split}_true"], out[f"{split}_obs"] = true, obs
        out[f"{split}_T0"] = np.asarray(d[f"{split}_T"], dtype=np.float64)
        out[f"{split}_branch_raw"] = np.column_stack(
            [true[:, 0, BRANCH_SPECIES[0]], true[:, 0, BRANCH_SPECIES[1]], out[f"{split}_T0"]])
    return out


def author_global_stats(data: dict) -> dict:
    """ymin/ymax per species over the Julia ``ode_data_list`` (noisy train, noisy test,
    noise-free test; every time point) and the temperature range of all initial states."""
    ode_data_list = np.concatenate([data["train_obs"], data["test_obs"], data["test_true"]])
    all_T = np.concatenate([data["train_T0"], data["test_T0"]])
    return {"ymin": ode_data_list.min(axis=(0, 1)), "ymax": ode_data_list.max(axis=(0, 1)),
            "Tmin": float(all_T.min()), "Tmax": float(all_T.max()),
            "source": "ChemKAN_biodiesel_example.jl: species min/max over ode_data_list "
                      "(train obs + test obs + noise-free test, all times); T min/max over "
                      "all initial temperatures"}


def to_u(states, stats: dict):
    """Physical six-species states -> author-global normalized u (numpy or torch)."""
    ymin, ymax = stats["ymin"], stats["ymax"]
    if isinstance(states, torch.Tensor):
        ymin, ymax = (torch.as_tensor(v, dtype=states.dtype, device=states.device)
                      for v in (ymin, ymax))
    return (states - ymin) / (ymax - ymin)


def from_u(u, stats: dict):
    ymin, ymax = stats["ymin"], stats["ymax"]
    if isinstance(u, torch.Tensor):
        ymin, ymax = (torch.as_tensor(v, dtype=u.dtype, device=u.device) for v in (ymin, ymax))
    return u * (ymax - ymin) + ymin


def branch_input(branch_raw: np.ndarray, mode: str, stats: dict) -> np.ndarray:
    """[TG0, ROH0, T0] in the representation of ``mode``."""
    if mode == "raw_states":
        return branch_raw.copy()
    if mode == "author_global_normalized_states":
        s = list(BRANCH_SPECIES)
        out = branch_raw.copy()
        out[:, :2] = (branch_raw[:, :2] - stats["ymin"][s]) / (stats["ymax"][s] - stats["ymin"][s])
        out[:, 2] = (branch_raw[:, 2] - stats["Tmin"]) / (stats["Tmax"] - stats["Tmin"])
        return out
    raise ValueError(f"unknown state mode {mode!r}; expected one of {STATE_MODES}")


# Branch-only preprocessing ablation (targets stay in the state mode's representation).
BRANCH_MODES = ("follow_state_mode", "normalized", "raw", "raw_species_norm_T")


def branch_input_mode(branch_raw: np.ndarray, branch_mode: str, state_mode: str,
                      stats: dict) -> np.ndarray:
    """[TG0, ROH0, T0] under ``branch_mode``: follow_state_mode (default, previous
    behaviour), normalized (A: all three author-global normalized), raw (B: physical
    values), raw_species_norm_T (C: TG0/ROH0 physical, T0 normalized)."""
    if branch_mode == "follow_state_mode":
        return branch_input(branch_raw, state_mode, stats)
    norm = branch_input(branch_raw, "author_global_normalized_states", stats)
    if branch_mode == "normalized":
        return norm
    if branch_mode == "raw":
        return branch_raw.copy()
    if branch_mode == "raw_species_norm_T":
        out = branch_raw.copy()
        out[:, 2] = norm[:, 2]
        return out
    raise ValueError(f"unknown branch mode {branch_mode!r}; expected one of {BRANCH_MODES}")


def encode_states(states: np.ndarray, mode: str, stats: dict) -> np.ndarray:
    """Six-species states in the representation the network predicts under ``mode``."""
    if mode == "raw_states":
        return states.copy()
    if mode == "author_global_normalized_states":
        return to_u(states, stats)
    raise ValueError(f"unknown state mode {mode!r}; expected one of {STATE_MODES}")


def decode_states(pred, mode: str, stats: dict):
    """Network output -> physical states."""
    return pred if mode == "raw_states" else from_u(pred, stats)


# ---- metrics ---------------------------------------------------------------------------

def _check(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"expected [trajectory, time, species]; got {tuple(pred.shape)}")


def mean_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ordinary MSE over trajectories, times and species."""
    _check(pred, target)
    return (pred - target).square().mean()


def eq18(pred_u: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
    """Eq. 18 reduction on normalized states: mean over species, SUM over time (no
    division by the number of time points), mean over trajectories (Julia's
    total / n_exp_train aggregation)."""
    _check(pred_u, target_u)
    per_traj = (pred_u - target_u).square().mean(dim=SPECIES_DIM).sum(dim=TIME_DIM)
    return per_traj.mean()


def all_metrics(pred_rep: torch.Tensor, obs_rep: torch.Tensor, pred_u: torch.Tensor,
                obs_u: torch.Tensor, true_u: torch.Tensor) -> dict:
    """``*_rep``: the trained representation; ``*_u``: author-global normalized units."""
    n_t = pred_u.shape[TIME_DIM]
    out = {"mean_mse": float(mean_mse(pred_rep, obs_rep)),
           "mean_mse_u": float(mean_mse(pred_u, obs_u)),
           "eq18_obs": float(eq18(pred_u, obs_u)),
           "eq22_true": float(eq18(pred_u, true_u))}
    out["eq18_obs_time_averaged"] = out["eq18_obs"] / n_t
    out["eq22_true_time_averaged"] = out["eq22_true"] / n_t
    return out


def objective_fn(name: str):
    """(pred_rep, obs_rep, pred_u, obs_u, true_u) -> scalar loss to optimize."""
    if name == "mean_mse":
        return lambda pr, orep, pu, ou, tu: mean_mse(pr, orep)
    if name == "eq18_obs":
        return lambda pr, orep, pu, ou, tu: eq18(pu, ou)
    if name == "eq22_true":
        return lambda pr, orep, pu, ou, tu: eq18(pu, tu)
    raise ValueError(f"unknown objective {name!r}; expected one of {OBJECTIVES}")


# ---- initialization ablation -----------------------------------------------------------
# Verified definitions (sources read 2026-10-01):
#   DeepXDE 0.11.2 "Glorot normal" = tf.compat.v1.glorot_normal_initializer =
#     VarianceScaling(scale=1, mode="fan_avg", distribution="truncated_normal"):
#     N(0, s0) with s0 = sqrt(2 / (fan_in + fan_out)) / 0.87962566103423978, truncated at
#     +-2 s0 (post-truncation std = sqrt(2 / (fan_in + fan_out))); tf.layers.dense biases
#     default to zeros. DeepXDE is reference-implementation evidence, NOT the confirmed
#     ChemKAN DeepONet code.
#   Lux 1.31.4 Dense defaults (framework of the released ChemKAN code; not confirmed for
#     the unpublished DeepONet): weight U(+-sqrt(3) * gain / sqrt(fan_in)) with
#     gain = Lux.Utils.calculate_gain(activation) = 2 for relu, 1 for identity;
#     bias U(+-1 / sqrt(fan_in)).
#   PyTorch 2.13 nn.Linear defaults: weight U(+-1 / sqrt(fan_in)) (kaiming_uniform_,
#     a = sqrt(5)); bias U(+-1 / sqrt(fan_in)).
INIT_SCHEMES = {
    "A_xavier_normal_zero_bias": ("glorot_normal_pytorch_equivalent", "zeros", "baseline"),
    "Aprime_tf_truncated_glorot_zero_bias": ("glorot_normal_tf_truncated", "zeros",
                                             "exact DeepXDE/TF1 sampling"),
    "C_xavier_normal_uniform_bias": ("glorot_normal_pytorch_equivalent",
                                     "uniform(+-1/sqrt(fan_in)) (Lux = PyTorch rule)",
                                     "bias-only ablation"),
    "R3_glorot_uniform_zero_bias": ("glorot_uniform (xavier_uniform rule, "
                                    "U(+-sqrt(6/(fan_in+fan_out))))", "zeros",
                                    "Glorot-uniform framework-default candidate"),
    "B_lux_default": ("lux_1.31.4_default kaiming_uniform(gain=calculate_gain(act))",
                      "uniform(+-1/sqrt(fan_in))", "framework-default ablation (Lux)"),
    "B_torch_default": ("pytorch_default kaiming_uniform(a=sqrt(5))",
                        "uniform(+-1/sqrt(fan_in))", "framework-default ablation (PyTorch)"),
}


def _linear_layers(model):
    """(Linear, followed_by_relu) for every Linear of branch, trunk and head."""
    out = []
    for seq in (model.branch, model.trunk):
        mods = list(seq)
        for i, m in enumerate(mods):
            if isinstance(m, torch.nn.Linear):
                out.append((m, i + 1 < len(mods) and isinstance(mods[i + 1], torch.nn.ReLU)))
    out.append((model.head, False))
    return out


def apply_init(model, scheme: str, seed: int):
    """Re-initialize ``model`` (built by ``biodiesel_deeponet.build(seed=seed)``, i.e. the
    baseline xavier_normal_ weights and zero biases) according to ``scheme``.

    C keeps the baseline weights bit-for-bit and draws only biases; A' and B redraw weights.
    Extra draws use a generator seeded from ``seed`` so every arm is deterministic.
    """
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"unknown init scheme {scheme!r}; expected one of {list(INIT_SCHEMES)}")
    if scheme == "A_xavier_normal_zero_bias":
        return model
    g = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for lin, relu_after in _linear_layers(model):
            fan_out, fan_in = lin.weight.shape
            b_bound = 1.0 / np.sqrt(fan_in)
            if scheme == "Aprime_tf_truncated_glorot_zero_bias":
                s0 = np.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
                w = torch.randn(lin.weight.shape, generator=g)
                bad = w.abs() > 2.0
                while bad.any():                      # TF truncated_normal: redraw beyond 2 sd
                    w[bad] = torch.randn(int(bad.sum()), generator=g)
                    bad = w.abs() > 2.0
                lin.weight.copy_(w * s0)
                lin.bias.zero_()
            elif scheme == "R3_glorot_uniform_zero_bias":
                lim = np.sqrt(6.0 / (fan_in + fan_out))
                lin.weight.copy_((torch.rand(lin.weight.shape, generator=g) - 0.5) * 2 * lim)
                lin.bias.zero_()
            elif scheme == "C_xavier_normal_uniform_bias":
                lin.bias.copy_((torch.rand(lin.bias.shape, generator=g) - 0.5) * 2 * b_bound)
            else:
                if scheme == "B_lux_default":
                    w_bound = np.sqrt(3.0) * (2.0 if relu_after else 1.0) / np.sqrt(fan_in)
                else:
                    w_bound = 1.0 / np.sqrt(fan_in)
                lin.weight.copy_((torch.rand(lin.weight.shape, generator=g) - 0.5) * 2 * w_bound)
                lin.bias.copy_((torch.rand(lin.bias.shape, generator=g) - 0.5) * 2 * b_bound)
    return model


# ---- constructive (planted-kink) diagnostic -- NOT a source-faithful setting ------------

def plant_knots(model, knots_s, t_end: float, layer2_bias: float | None = None) -> dict:
    """DIAGNOSTIC ONLY. First-layer trunk units 0..K-1 become ReLU((t - k)/t_end): hinge at
    the physical knot k, slope 1/t_end so the activation stays in [0, 1] on 0..t_end (raw t
    input). Units K..6 keep their initialization draw untouched. ``layer2_bias`` (artificial
    masking diagnostic) sets every second-layer trunk bias to that value."""
    lin = model.trunk[0]
    if len(knots_s) > lin.out_features:
        raise ValueError("more knots than first-layer trunk units")
    with torch.no_grad():
        for i, k in enumerate(knots_s):
            lin.weight[i, 0] = 1.0 / t_end
            lin.bias[i] = -float(k) / t_end
        if layer2_bias is not None:
            model.trunk[2].bias.fill_(float(layer2_bias))
    return {"planted_units": list(range(len(knots_s))), "knots_s": [float(k) for k in knots_s],
            "weight": 1.0 / t_end, "untouched_units": list(range(len(knots_s), lin.out_features)),
            "layer2_bias": layer2_bias}


def branch_stats(model, branch_in: torch.Tensor) -> dict:
    """Pre-activation statistics of the two hidden branch layers on the given inputs."""
    out, x = {}, branch_in
    with torch.no_grad():
        for li, (lin, act) in enumerate(((model.branch[0], model.branch[1]),
                                         (model.branch[2], model.branch[3]))):
            z = lin(x)
            a = act(z)
            out[f"layer{li + 1}"] = {"pre_min": float(z.min()), "pre_max": float(z.max()),
                                     "pre_mean": float(z.mean()), "pre_std": float(z.std()),
                                     "frac_relu_zero": float((a == 0).float().mean()),
                                     "dead_units": int((a.abs().max(dim=0).values == 0).sum())}
            x = a
    return out


def useful_breakpoints(model, branch_in: torch.Tensor, t_end: float, threshold: float = 1e-3,
                       n: int = 3001) -> dict:
    """Interior (0, t_end) trunk breakpoints whose unit's downstream contribution exceeds
    ``threshold`` (normalized output units), raw-t input.

    Layer-2 unit j: max_{t, traj, species} |head_W[s, j] * branch[n, j] * a2_j(t)|.
    Layer-1 unit i: max over t, traj, species, j of
        |head_W[s, j] * branch[n, j] * 1[z2_j(t) > 0] * W2[j, i] * a1_i(t)|.
    """
    t = torch.linspace(0.0, t_end, n).reshape(-1, 1)
    with torch.no_grad():
        z1 = model.trunk[0](t); a1 = torch.relu(z1)
        z2 = model.trunk[2](a1); a2 = torch.relu(z2)
        b = model.branch(branch_in)                                  # (N, 8)
        H = model.head.weight                                        # (6, 8)
        coef = (H.abs().unsqueeze(1) * b.abs().unsqueeze(0)).amax(dim=(0, 1))   # (8,) max_s,n
        contrib2 = coef * a2.abs().amax(dim=0)                       # (8,)
        gate = (z2 > 0).float()                                      # (n, 8)
        W2 = model.trunk[2].weight                                   # (8, 7)
        contrib1 = torch.zeros(W2.shape[1])
        for i in range(W2.shape[1]):
            path = gate * (W2[:, i].abs() * coef).unsqueeze(0) * a1[:, i:i + 1].abs()
            contrib1[i] = path.max()
    bp = interior_breakpoints(model, t_end)
    w1 = model.trunk[0].weight.detach().ravel().numpy()
    b1 = model.trunk[0].bias.detach().ravel().numpy()
    l1 = []
    for i in range(len(w1)):
        if w1[i] != 0:
            k = -b1[i] / w1[i]
            if 0 < k < t_end:
                l1.append((float(k), i, float(contrib1[i])))
    tt = t.ravel().numpy()
    l2 = []
    for j in range(z2.shape[1]):
        sgn = np.sign(z2[:, j].numpy())
        for idx in np.flatnonzero(sgn[1:] * sgn[:-1] < 0) + 1:
            if 0 < tt[idx] < t_end:
                l2.append((float(tt[idx]), j, float(contrib2[j])))
    useful = [p for p, _, c in l1 + l2 if c > threshold]
    return {"threshold": threshold,
            "layer1": [{"t": p, "unit": u, "contribution": c} for p, u, c in sorted(l1)],
            "layer2": [{"t": p, "unit": u, "contribution": c} for p, u, c in sorted(l2)],
            "useful_positions_s": sorted(useful), "n_useful": len(useful),
            "n_interior": len(bp["layer1"]) + len(bp["layer2"])}


# ---- trunk diagnostics -----------------------------------------------------------------

def raw_trunk_time(data: dict, data_file=DEFAULT_DATA) -> torch.Tensor:
    """The trunk input: the repository's physical time grid, unscaled (float32). Asserts it
    equals the archive's ``t`` exactly; no configuration can rescale it."""
    t = torch.as_tensor(data["t"], dtype=torch.float32)
    repo = torch.as_tensor(np.load(data_file)["t"], dtype=torch.float32)
    assert torch.equal(t, repo), "trunk time differs from the repository time grid"
    assert float(t.min()) == float(repo.min()) and float(t.max()) == float(repo.max())
    return t


def trunk_activations(model, t: torch.Tensor) -> dict:
    """Per-layer trunk activations on the time points ``t`` (T,), as numpy (T, width)."""
    x = t.reshape(-1, 1)
    out, i = {}, 0
    for layer in model.trunk:
        x = layer(x)
        if isinstance(layer, (torch.nn.ReLU, torch.nn.Tanh)):
            name = "relu" if isinstance(layer, torch.nn.ReLU) else "tanh"
            out[f"{name}{i}"] = x.detach().cpu().numpy()
            i += 1
    return out


ACTIVATIONS = ("relu", "tanh")


def set_activation(model, name: str):
    """Swap every hidden ReLU of branch and trunk for ``name`` in place. Positions and
    parameters are unchanged (final branch layer and head stay linear; the final trunk
    layer stays activated), so the parameter tensors are identical across activations."""
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}")
    if name == "relu":
        return model
    for seq in (model.branch, model.trunk):
        for i, m in enumerate(seq):
            if isinstance(m, torch.nn.ReLU):
                seq[i] = torch.nn.Tanh()
    return model


PLACEMENTS = {
    "A": {"branch_final": "linear", "trunk_final": "activated",
          "label": "Lu-stated / DeepXDE-reference placement; primary source-supported reconstruction"},
    "B": {"branch_final": "linear", "trunk_final": "linear",
          "label": "conventional hidden-only MLP interpretation; plausible, unconfirmed"},
    "C": {"branch_final": "activated", "trunk_final": "activated",
          "label": "activation after every branch/trunk affine layer; plausible, unconfirmed"},
    "D": {"branch_final": "activated", "trunk_final": "linear",
          "label": "mixed alternative; plausible, unconfirmed"},
}


def set_placement(model, placement: str):
    """Final-latent activation placement on a model built with placement A (branch final
    linear, trunk final activated). Only activation modules change: parameters and state_dict
    keys are identical across A-D, so paired runs can share initial tensors exactly."""
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}")
    if not isinstance(model.trunk[-1], (torch.nn.ReLU, torch.nn.Tanh)):
        raise ValueError("set_placement expects the default placement-A model")
    act = type(model.trunk[-1])
    if PLACEMENTS[placement]["trunk_final"] == "linear":
        model.trunk = torch.nn.Sequential(*list(model.trunk)[:-1])
    if PLACEMENTS[placement]["branch_final"] == "activated":
        model.branch = torch.nn.Sequential(*list(model.branch), act())
    return model


def tanh_saturation(model, t_in: torch.Tensor) -> dict:
    """Pre-activation / saturation statistics of both tanh trunk layers on ``t_in``."""
    out, x = {}, t_in.reshape(-1, 1)
    with torch.no_grad():
        for li, (lin, act) in enumerate(((model.trunk[0], model.trunk[1]),
                                         (model.trunk[2], model.trunk[3]))):
            z = lin(x)
            a = act(z)
            az, aa = z.abs().numpy().ravel(), a.abs().numpy().ravel()
            d = (1 - a ** 2).numpy().ravel()
            out[f"layer{li + 1}"] = {
                "abs_z_quantiles_10_50_90_max": [float(np.quantile(az, q)) for q in (0.1, 0.5, 0.9)] + [float(az.max())],
                "frac_abs_z_gt": {str(c): float((az > c).mean()) for c in (2, 3, 5)},
                "frac_abs_act_gt": {str(c): float((aa > c).mean()) for c in (0.95, 0.99)},
                "derivative_mean": float(d.mean()), "derivative_median": float(np.median(d))}
            x = a
    return out


def first_layer_kinks(model) -> np.ndarray:
    """t at which each first-layer trunk ReLU switches (w t + b = 0); nan when w = 0."""
    lin = model.trunk[0]
    w = lin.weight.detach().cpu().numpy().ravel()
    b = lin.bias.detach().cpu().numpy().ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w != 0, -b / w, np.nan)


def interior_breakpoints(model, t_end: float, n: int = 30001) -> dict:
    """Where trunk ReLUs switch inside the open interval 0 < t < t_end, in seconds.

    layer1: kinks t = -b/w of the first-layer units; layer2: sign changes of the
    second-layer pre-activations on a dense grid (the final trunk ReLU is the only place
    the second layer adds breakpoints). A layer-2 location is the first dense point after
    the sign change, so it is exact to t_end / (n - 1). The trunk input is raw time, so
    positions are in physical seconds. A trunk without a final activation has no layer-2
    breakpoints.
    """
    t = torch.linspace(0.0, t_end, n)
    with torch.no_grad():
        h1 = model.trunk[1](model.trunk[0](t.reshape(-1, 1)))
        z2 = model.trunk[2](h1).numpy()
    tt = t.numpy()
    layer2 = []
    for j in range(z2.shape[1]):
        s = np.sign(z2[:, j])
        idx = np.flatnonzero(s[1:] * s[:-1] < 0) + 1
        layer2 += [float(tt[i]) for i in idx if 0.0 < tt[i] < t_end]
    if len(model.trunk) < 4:                              # final trunk layer linear
        layer2 = []
    k = first_layer_kinks(model)
    layer1 = [float(x) for x in k if np.isfinite(x) and 0.0 < x < t_end]
    return {"layer1": sorted(layer1), "layer2": sorted(layer2)}


def linearity_index(pred: np.ndarray, truth: np.ndarray, t: np.ndarray) -> float:
    """Curvature of the prediction relative to the truth's, [traj, time, species].

    For every trajectory and species: RMS of (series - its least-squares straight line in
    t), prediction over truth; the median over all series. 0 = straight lines; ~1 = as
    much curvature as the true trajectories.
    """
    A = np.column_stack([t, np.ones_like(t)])

    def curvature(y):                                   # y: (time, series)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.sqrt(((y - A @ coef) ** 2).mean(axis=0))

    p = pred.transpose(1, 0, 2).reshape(len(t), -1)
    q = truth.transpose(1, 0, 2).reshape(len(t), -1)
    cq = curvature(q)
    keep = cq > 1e-8
    return float(np.median(curvature(p)[keep] / cq[keep]))


def dead_dimensions(acts: dict) -> dict:
    """Units that are zero at every evaluated time point, per trunk ReLU layer."""
    return {k: int((np.abs(a).max(axis=0) == 0).sum()) for k, a in acts.items()}


def prediction_stats(pred_phys: torch.Tensor) -> dict:
    """Summary of physical predictions [traj, time, species]."""
    p = pred_phys.detach().cpu().numpy()
    return {"min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()),
            "abs_max": float(np.abs(p).max()),
            "per_species_min": p.min(axis=(0, 1)).tolist(),
            "per_species_max": p.max(axis=(0, 1)).tolist(),
            "fraction_negative": float((p < 0).mean())}
