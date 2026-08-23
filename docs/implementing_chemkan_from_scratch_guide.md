# Implementing ChemKAN from scratch (a learning guide)

A step-by-step path to building the ChemKAN model in PyTorch **by hand**, so you
understand every piece. Each milestone follows the same rhythm:

> **Intuition** → **the paper's math** → **code you write** → **a check that proves you understood it**

You already generated the training data (`chemkan/data/generated/*.npz` — see the
[data-generation walkthrough](../chemkan/scripts/data_gen/data_generation_walkthrough.ipynb)).
This guide builds the *model* that learns from it.

Paper: *ChemKANs for Combustion Chemistry Modeling and Acceleration*, Koenig, Kim
& Deng, 2025 ([`docs/paper/…pdf`](paper/ChemKANs_for_Combustion_Chemistry_Modeling_and_Acceleration.pdf),
[arXiv:2504.12580](https://arxiv.org/pdf/2504.12580)). Equation numbers below refer to it.

---

## How to use this guide

- **Do the milestones in order.** Each one is runnable on its own and ends in a
  plot or a number you can sanity-check. Don't move on until the check passes.
- **Write the code yourself first.** Every milestone gives you a skeleton with
  the shapes worked out and the hard line(s) hinted. A `<details>` "reference"
  block holds a working version — open it *after* you've tried.
- **Keep the paper open at Fig. 2 and Eqs. 13–17.** That one figure is the whole
  architecture.

### Which baselines, and how to use them (do this before Milestone 1)

| Repo | How you use it | Notes |
|---|---|---|
| [pykan](https://github.com/KindXiaoming/pykan) | Run the "Hello, KAN" and "fit a formula" intro notebooks once. **Watch an activation function get learned.** Do not copy its code. | B-spline basis; heavy. This is for *intuition only*. |
| [DENG-MIT/LeanKAN](https://github.com/DENG-MIT/LeanKAN), [DENG-MIT/KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs) | Your **canonical reference**. Same authors as ChemKAN. When your shapes/loss don't match the paper, compare here. The repo's [`Lotka-Volterra-Pytorch/predator_prey.py`](https://github.com/DENG-MIT/KAN-ODEs/blob/main/Lotka-Volterra-Pytorch/predator_prey.py) is a ready-made **PyTorch KAN-ODE training loop** — borrow it in Milestone 4. | Most of the repo is Julia; the *math* is language-independent. The one Python example uses a **B-spline** layer (`efficientkan.py`) — reuse its *loop*, not its *layer*. |
| Your [`chemkan/src/addkan/addkan.py`](../chemkan/src/addkan/addkan.py) (= [efficient-kan](https://github.com/Blealtan/efficient-kan)) | Read `AddKANLinear.forward` once to see how a KAN layer is wired. | **B-spline**, not RBF. ChemKAN uses RBF (Eq. 11–12), so you will *not* reuse this activation. Good structural reference, wrong basis. |

**The single most important fact:** the paper's activation is a **Gaussian RBF
sum** (Eq. 11–12), and the layers are **LeanKAN** (multiply + add, Eq. 8–10) —
*not* B-splines and *not* addition-only. That's what you'll build.

### Environment

```bash
# add to requirements-dev.txt (model side; keep the data-gen runtime pins as-is)
pip install torch torchdiffeq matplotlib
```

`torchdiffeq` gives you the differentiable ODE solver for the Neural-ODE part
(Milestone 4+). Everything before that is plain `torch`.

---

## The one idea behind KANs (read this once)

An **MLP** layer:  `y_i = σ( Σ_j  w_ij · x_j )` — *fixed* activation `σ`,
*learnable scalars* `w_ij` on the edges.

A **KAN** layer:   `y_i = Σ_j  φ_ij( x_j )` — a *learnable function* `φ_ij` on
every edge, then a plain sum at the node.

> **Learnable activation functions on the edges; addition at the nodes.**

Why it's plausible: the Kolmogorov–Arnold theorem says any continuous
multivariate function can be written as sums of *univariate* functions. KANs
parameterize exactly those univariate functions. For chemistry this is a natural
fit — a reaction rate really *is* a univariate function of temperature
(Arrhenius, `exp(−Eₐ/RT)`) combined with products of concentrations.

Everything below is: (1) make one learnable univariate `φ`, (2) tile them into a
layer, (3) add multiplication (LeanKAN), (4) make the layer a *derivative* and
integrate it (KAN-ODE), (5) wire the layers to mimic the physics (ChemKAN),
(6) train in two stages.

---

## Milestone 1 — one learnable univariate function φ(x)

**Goal:** build a single edge activation and *watch it learn an arbitrary shape.*
This is the atom of the whole network.

### Math (Eq. 11–12)

$$\phi_{l,\alpha,\beta}(\mathbf{x}) = \sum_{i=1}^{N} w^{\psi}_{l,\alpha,\beta,i}\,\psi(\lVert \mathbf{x} - c_i\rVert) \;+\; w^{b}_{l,\alpha,\beta}\, b(\mathbf{x}),
\qquad \psi(r) = \exp\!\left(-\frac{r^2}{2h^2}\right)$$

- `φ_{l,α,β}` — the activation on the edge of layer `l` connecting **input node `β`
  to output node `α`** (the paper's example: `α=3, β=2` connects the 2nd input to
  the 3rd output). A single edge for now; the `l,α,β` become tensor dimensions in
  Milestone 2.
- `i` — index over the `N` basis functions here (**not** an output index); `c_i`
  are the `N` fixed grid centers (a buffer, not trained).
- `h` — grid spacing (controls bump width).
- `ψ` — a Gaussian bump around each center.
- `w^ψ_{l,α,β,i}` — **learnable** height of basis `i` on edge `(α,β)`. These bend the curve into any smooth shape.
- `b(x)` — a fixed base activation, **Swish/SiLU** in the paper, with learnable weight `w^b_{l,α,β}` (a residual path so `φ` isn't stuck at 0 early on).

**Intuition:** you're building a curve out of `N` Gaussian bumps. Raising bump `i`
lifts the curve near `c_i`. With enough bumps you can trace any smooth 1-D function.

### Code you write

```python
import torch, torch.nn as nn

class RBFActivation(nn.Module):
    """One learnable univariate function phi_{l,a,b}(x)  (paper Eq. 11-12).

    This is a SINGLE edge, so the (l, alpha, beta) subscripts are fixed and
    dropped; in Milestone 2 they become the (layer, out, in) tensor dimensions.
    """
    def __init__(self, num_basis=10, x_min=-1.0, x_max=1.0):
        super().__init__()
        centers = torch.linspace(x_min, x_max, num_basis)     # c_i  (fixed)
        self.register_buffer("centers", centers)
        self.h = (x_max - x_min) / (num_basis - 1)            # spacing h
        self.w_rbf  = nn.Parameter(torch.randn(num_basis) * 0.1)  # w^psi_{...,i} (learnable)
        self.w_base = nn.Parameter(torch.zeros(1))               # w^b           (learnable)
        self.base   = nn.SiLU()                                  # b(x) = Swish

    def forward(self, x):                     # x: (batch,)
        # TODO 1: r[b,i] = x[b] - c_i          -> shape (batch, num_basis)
        # TODO 2: psi = exp(-r^2 / (2 h^2))    -> Gaussian bumps  (Eq. 12)
        # TODO 3: rbf = sum_i w_rbf[i] * psi[b,i]
        # TODO 4: return rbf + w_base * base(x)
        ...
```

<details><summary>Reference forward</summary>

```python
    def forward(self, x):                     # x: (batch,)
        r   = x.unsqueeze(-1) - self.centers          # (batch, num_basis)
        psi = torch.exp(-(r ** 2) / (2 * self.h ** 2))# (batch, num_basis)
        rbf = psi @ self.w_rbf                        # (batch,)  sum over bumps
        return rbf + self.w_base * self.base(x)
```
</details>

### Check — fit φ to a target and *look at it*

```python
phi = RBFActivation(num_basis=12)
opt = torch.optim.Adam(phi.parameters(), lr=0.05)
x = torch.linspace(-1, 1, 200)
y = torch.sin(3 * x)                     # any smooth target

for _ in range(2000):
    opt.zero_grad()
    loss = ((phi(x) - y) ** 2).mean()
    loss.backward(); opt.step()
print("final MSE:", loss.item())         # expect ~1e-4 or lower

import matplotlib.pyplot as plt
plt.plot(x, y, "k", label="target")
plt.plot(x.detach(), phi(x).detach(), "r--", label="learned phi")
# bonus: plot each bump  w_rbf[i]*psi_i  to SEE the decomposition
plt.legend(); plt.show()
```

**You understood it if:** the red curve overlaps the target, and lowering
`num_basis` to 3 visibly hurts the fit (too few bumps). That knob is the paper's
grid size `N`.

---

## Milestone 2 — the AddKAN layer (Eq. 7)

**Goal:** tile `φ`'s into a layer that maps `in_features → out_features`.

### Math (Eq. 7 = the activation matrix; Eq. 10 = the additive node)

Eq. 7 defines the AddKAN layer as the **matrix of activation functions** `Φ_l`,
of size `n_{l+1} × n_l`:

$$\text{AddKAN}:\ \Psi_l^{\text{add}} = \Phi_l =
\begin{pmatrix}
\phi_{l,1,1}(\cdot) & \phi_{l,1,2}(\cdot) & \cdots & \phi_{l,1,n_l}(\cdot)\\
\phi_{l,2,1}(\cdot) & \phi_{l,2,2}(\cdot) & \cdots & \phi_{l,2,n_l}(\cdot)\\
\vdots & \vdots & & \vdots\\
\phi_{l,n_{l+1},1}(\cdot) & \phi_{l,n_{l+1},2}(\cdot) & \cdots & \phi_{l,n_{l+1},n_l}(\cdot)
\end{pmatrix}$$

The **node operation** — how output `i` is actually formed — is the pure-addition
case of LeanKAN, i.e. Eq. 10 with `n_l^{mu} = 0`:

$$y_{l,i}^{\text{add}} = \sum_{j=1}^{n_l} \phi_{l,i,j}(x_{l,j}),
\qquad i \in \{1, 2, \dots, n_{l+1}\}$$

One learnable `φ_{l,i,j}` per (output node `i`, input node `j`) edge — the
`n_{l+1} × n_l` matrix above. Output node `i` sums its edges. Total: `n_l · n_{l+1}`
activations (compare an MLP: `n_l · n_{l+1}` scalar weights).

Two paper details:
- **tanh input normalization.** The RBF grid lives on `[-1, 1]`; the paper passes
  each layer input through `tanh` first so values stay on the grid (instead of
  re-gridding). Do `x = torch.tanh(x)` at the top of `forward`.
- The base activation `b(x)` is shared as Swish per edge.

### Shapes (this is where beginners get lost — internalize it)

```
x            : (B, in)
tanh(x)      : (B, in)
r            : (B, in, N)        # x minus each center
psi          : (B, in, N)        # Gaussian bumps per input, per center
w_rbf        : (out, in, N)      # learnable bump heights per edge
w_base       : (out, in)         # learnable base weight per edge
y            : (B, out)          # sum over in AND over N
```

The whole layer is two contractions: `psi · w_rbf` summed over `(in, N)`, plus
`base(x) · w_base` summed over `in`. `einsum` makes this one line each.

### Code you write

```python
class AddKANLayer(nn.Module):
    """KAN layer with additive nodes (paper Eq. 7), RBF activations."""
    def __init__(self, in_features, out_features, num_basis=8, grid=(-1.0, 1.0)):
        super().__init__()
        c = torch.linspace(grid[0], grid[1], num_basis)
        self.register_buffer("centers", c)
        self.h = (grid[1] - grid[0]) / (num_basis - 1)
        self.w_rbf  = nn.Parameter(torch.randn(out_features, in_features, num_basis) * 0.1)
        self.w_base = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base   = nn.SiLU()

    def forward(self, x):                       # x: (B, in)
        x   = torch.tanh(x)                     # keep on the grid  (paper detail)
        r   = x.unsqueeze(-1) - self.centers    # (B, in, N)
        psi = torch.exp(-(r ** 2) / (2 * self.h ** 2))   # (B, in, N)
        # TODO: rbf  = sum over (in, N) of  psi * w_rbf   -> (B, out)
        # TODO: base = sum over  in     of  base(x)*w_base-> (B, out)
        # hint: torch.einsum("bik,oik->bo", psi, self.w_rbf)
        ...
```

<details><summary>Reference forward</summary>

```python
    def forward(self, x):
        x   = torch.tanh(x)
        r   = x.unsqueeze(-1) - self.centers
        psi = torch.exp(-(r ** 2) / (2 * self.h ** 2))
        rbf  = torch.einsum("bik,oik->bo", psi, self.w_rbf)          # (B, out)
        base = torch.einsum("bi,oi->bo", self.base(x), self.w_base)  # (B, out)
        return rbf + base
```
</details>

### Check

- Fit a 2-input function, e.g. `f(x₁,x₂) = sin(πx₁)·x₂`, with
  `AddKANLayer(2, 8) → AddKANLayer(8, 1)`. MSE should drop well below 1e-2.
- Count parameters and confirm it scales as `out·in·(N+1)`.
- Sanity vs. reference: this is the RBF analogue of `AddKANLinear.forward` in your
  `addkan.py` — same wiring, Gaussian bumps instead of B-splines.

---

## Milestone 3 — the LeanKAN layer (Eq. 8–10)

**Goal:** add multiplication. This is the layer ChemKAN actually uses.

### Why (intuition first)

Addition-only KANs are **bad at products**. But chemistry is full of products:
mass-action rates are `k·[A]·[B]`. To make an AddKAN produce `x₁·x₂` it has to
approximate it through sums of bumps — expensive and imprecise. LeanKAN adds a
cheap **multiplication sublayer** so products are represented directly. That's a
big part of why ChemKAN hits **344 parameters** for H₂ — it's *parameter-lean*.

### Math (Eq. 8–10)

LeanKAN is the **sum of a multiplication term and an addition term** (Eq. 8):

$$\Psi_l^{\text{lean}}(\mathbf{x}_l) = \mathbf{y}_l^{\text{mult}} + \mathbf{y}_l^{\text{add}} \in \mathbb{R}^{n_{l+1}}$$

The first `n_l^{mu}` inputs feed a **product** (Eq. 9); the remaining inputs feed a
**sum** (Eq. 10). For each output `i ∈ {1, 2, …, n_{l+1}}`:

$$y_{l,i}^{\text{mult}} = \prod_{j=1}^{n_l^{mu}} \phi_{l,i,j}(x_{l,j}),
\qquad
y_{l,i}^{\text{add}} = \sum_{j=n_l^{mu}+1}^{n_l} \phi_{l,i,j}(x_{l,j})$$

- `n_l^{mu}` — the **multiplication hyperparameter**: the number of multiplication
  input nodes for layer `l`.
- `φ_{l,i,j}` — the **same** univariate activations as AddKAN (Eq. 11–12).
- `n_l^{mu} = 0` ⟹ `y_l^{mult}` is empty and LeanKAN reduces to AddKAN (Eq. 7).
  (So you can implement one class.)

### Code you write

Reuse Milestone 2, but stop *before* summing over inputs — you need the per-edge
values `φ_{l,i,j}(x_{l,j})` so you can multiply some and add others.

```python
class LeanKANLayer(nn.Module):
    """LeanKAN: multiply the first n_mu inputs, add the rest (Eq. 8-10)."""
    def __init__(self, in_features, out_features, n_mu=0, num_basis=8, grid=(-1.,1.)):
        super().__init__()
        # ... same params as AddKANLayer ...
        self.n_mu = n_mu

    def edge_outputs(self, x):                  # -> (B, out, in): phi_{o,i}(x_i) per edge
        x   = torch.tanh(x)
        r   = x.unsqueeze(-1) - self.centers
        psi = torch.exp(-(r ** 2) / (2 * self.h ** 2))
        rbf  = torch.einsum("bik,oik->boi", psi, self.w_rbf)         # (B, out, in)
        base = torch.einsum("bi,oi->boi", self.base(x), self.w_base) # (B, out, in)
        return rbf + base

    def forward(self, x):
        g = self.edge_outputs(x)                # (B, out, in)
        # TODO: mult = product over the first n_mu inputs      (watch out below!)
        # TODO: add  = sum     over the remaining inputs
        # TODO: return mult + add
        ...
```

> **Watch out:** `g[..., :0].prod(dim=-1)` is an *empty product = 1*, not 0. When
> `n_mu == 0` you must set `mult = 0`, or you'll silently add a constant 1.

<details><summary>Reference forward</summary>

```python
    def forward(self, x):
        g = self.edge_outputs(x)                          # (B, out, in)
        mult = g[..., :self.n_mu].prod(dim=-1) if self.n_mu > 0 else 0.0
        add  = g[..., self.n_mu:].sum(dim=-1)
        return mult + add
```
</details>

### Check

- With `n_mu=0`, `LeanKANLayer` must give *identical* output to `AddKANLayer`
  (same seed). Assert it.
- Fit `f(x₁,x₂)=x₁·x₂` with a `LeanKANLayer(2, 1, n_mu=2)` and again with
  `n_mu=0`. The multiplicative one should fit dramatically better with fewer
  basis functions — that's the whole point of LeanKAN. Reference: DENG-MIT/LeanKAN.

---

## Milestone 4 — make it a derivative: KAN-ODE (Eq. 6)

**Goal:** stop predicting values; predict the *rate of change*, then integrate.

### Math (Eq. 6)

$$\frac{d\mathbf{u}}{dt} = \mathrm{KAN}(\mathbf{u}(t),\,\boldsymbol{\theta}) = (\Psi_{L-1} \circ \Psi_{L-2} \circ \cdots \circ \Psi_1 \circ \Psi_0)(\mathbf{u}(t))$$

The KAN is a composition of `L` layers `Ψ_0, …, Ψ_{L-1}` and it is the
**right-hand side** of an ODE. You get a trajectory by handing that
RHS to an ODE solver. Crucially, you train on the *integrated* trajectory (the
data you generated), and gradients flow **back through the solver** — that's the
Neural-ODE trick.

**Intuition:** instead of "given `t`, output the state" (that's DeepONet, Fig. 1A),
you learn the *local physics* "given the current state, how fast is it changing?"
— then a standard integrator marches it forward, exactly like a real kinetic solver.

> **Best reference for this milestone:** the authors' own PyTorch KAN-ODE demo,
> [`KAN-ODEs/Lotka-Volterra-Pytorch/predator_prey.py`](https://github.com/DENG-MIT/KAN-ODEs/blob/main/Lotka-Volterra-Pytorch/predator_prey.py).
> It confirms exactly this setup: `calDeriv(t, X): return model(X)` as the RHS,
> `torchdiffeq.odeint` (default solver = `dopri5`), MSE loss, `Adam(lr=2e-3)`.
> **Borrow its training-loop structure.** But note its layer file
> (`efficient_kan/efficientkan.py`, class `KANLinear`) is **B-spline + additive
> only** — the same efficient-kan as your `addkan.py`, *not* the RBF/LeanKAN that
> ChemKAN needs. So: use their **loop**, plug in **your** `AddKANLayer` from
> Milestone 2. (Their `regularization_loss`/`update_grid` are B-spline extras
> ChemKAN doesn't use — the paper replaces re-gridding with `tanh`.)

### Code you write

```python
from torchdiffeq import odeint

class KANODE(nn.Module):
    def __init__(self, kan): super().__init__(); self.kan = kan
    def forward(self, t, u):            # signature the solver wants: (scalar t, state u)
        return self.kan(u)              # du/dt = KAN(u);  autonomous -> ignore t

# roll out from initial states u0 over the saved time grid t
def rollout(func, u0, t):               # u0: (B, dim),  t: (T,)
    return odeint(func, u0, t, method="dopri5")   # -> (T, B, dim)
```

### Check — overfit ONE trajectory first (the classic Neural-ODE smoke test)

Load your biodiesel data and try to reproduce a *single* case before you touch the
whole set. Normalize with the **stored, train-only** `u_min/u_max` (you already
computed these in data-gen — reuse them; see `common.normalize`).

```python
import numpy as np
d = dict(np.load("chemkan/data/generated/biodiesel.npz"))
t   = torch.tensor(d["t"], dtype=torch.float32)                 # (30,)
u   = torch.tensor(d["train_states"], dtype=torch.float32)      # (20, 30, 6)
umin= torch.tensor(d["u_min"]); umax = torch.tensor(d["u_max"])
un  = (u - umin) / (umax - umin)                                # normalized to [0,1]

func = KANODE(AddKANLayer(6, 6))                # 6 species in/out (biodiesel, isothermal)
opt  = torch.optim.Adam(func.parameters(), lr=2e-3)             # paper's lr
case = un[0]                                                     # one trajectory (30, 6)
u0   = case[0:1]                                                 # (1, 6)
for step in range(3000):
    opt.zero_grad()
    pred = rollout(func, u0, t)[:, 0, :]        # (30, 6)
    loss = ((pred - case) ** 2).mean()
    loss.backward(); opt.step()
```

**You understood it if:** one trajectory is matched almost perfectly. If it blows
up (NaN), your ODE is too stiff for the explicit solver — see the box below.

> **Stiffness — the real difficulty of this project.** Combustion ODEs are stiff
> (H₂ especially). Two things tame it, both of which the paper does: (1) normalize
> **states** to `[0,1]` (done above), and (2) normalize **time** to `[0,1]`. Do both
> before integrating. The paper integrates with Tsit5 and uses **forward sensitivity
> analysis** (not adjoint) specifically for stiffness. `torchdiffeq`'s `dopri5` is
> the practical starting point; start with the *non-stiff* biodiesel case, get it
> working end-to-end, and only then take on H₂.

---

## Milestone 4½ — Lotka–Volterra warm-up (optional, recommended)

**Goal:** validate your KAN-ODE plumbing on a known-good, non-stiff problem
*before* fighting chemistry — and isolate "is my loop wrong?" from "is my layer wrong?".

Predator–prey is a smooth 2-variable ODE. The authors ship a full PyTorch demo
you can reproduce and check against their plots:
[`KAN-ODEs/Lotka-Volterra-Pytorch/`](https://github.com/DENG-MIT/KAN-ODEs/tree/main/Lotka-Volterra-Pytorch).

Do it in two passes:

1. **Run `predator_prey.py` as-is** (their B-spline `KANLinear`). If it reproduces
   their predator–prey curves, your `torchdiffeq` + environment are correct.
2. **Swap their layer for *your* `AddKANLayer`** (Milestone 2) — same 2-in/2-out
   KAN-ODE, RBF instead of B-spline. If it still fits, your layer is correct on a
   problem with a known answer.

Only then move to biodiesel, where any new failure is chemistry/stiffness — not
plumbing. (Their `predator_prey_adjoint.py` is the adjoint variant; it maps to the
paper's adjoint-vs-forward-sensitivity discussion. Adjoint is fine for non-stiff
LV, shaky for stiff H₂ — which is why the paper uses forward sensitivity.)

---

## Milestone 5 — the ChemKAN physics structure (Eq. 13–17, Fig. 2)

**Goal:** wire the layers to *mimic the governing equations* instead of being a
generic black box. This inductive bias is the paper's core contribution.

### Map the architecture to the physics (the key insight)

The real chemistry (Eq. 1–2):
- **Species rates** `dYᵢ/dt = (1/ρ)·Wᵢ·ω̇ᵢ` — a nonlinear function of the whole state.
- **Temperature rate** `dT/dt = −Σᵢ hᵢ Ẏᵢ / c_p` — note this is (almost) a *linear
  combination of the species rates* `Ẏᵢ`, weighted by `−hᵢ/c_p`.

ChemKAN encodes exactly that structure (Eq. 13–15):

$$\frac{d\bar{\mathbf u}}{dt} = \mathrm{KAN_{kin}}(\mathbf u,\theta_{kin})
\qquad
\frac{dT}{dt} = \underbrace{\mathrm{Linear}\!\left(\tfrac{d\bar{\mathbf u}}{dt}\right)}_{\text{the }-h_i/c_p\text{ weights}} + \underbrace{\mathrm{KAN_{cor}}(\mathbf u,\theta_{cor})}_{\text{small }c_p(\mathbf u)\text{ correction}}$$

- **Kinetic core** `KAN_kin`: input `u=[Y₁…Y_m,T]` (`m+1`), output the `m` species
  rates. Two layers (Eq. 16): `Ψ₁ˡᵉᵃⁿ ∘ Ψ₀ᵃᵈᵈ` — first **AddKAN**, then **LeanKAN**
  (`n_mu>0`) to inject the products chemistry needs.
- **Thermo superstructure** `dT/dt`: a **plain `nn.Linear` with no bias** over the
  species rates (that's the `−hᵢ/c_p` part, Eq. 14), **plus** a single-output
  **AddKAN** correction `KAN_cor` for the fact that `c_p` itself depends on the
  state (Eq. 15, 17).
- **Toggle:** biodiesel is isothermal → kinetics only (no thermo). H₂ → full model.

Look at Fig. 2: grey box = kinetic core, red box = thermo superstructure. You're
building exactly that.

### Code you write

```python
class ChemKAN(nn.Module):
    def __init__(self, m, hidden=None, n_mu=2, use_thermo=True, num_basis=8):
        super().__init__()
        hidden = hidden or (m + 1)
        # Kinetic core (Eq. 16): (m+1) -> hidden -> m   [AddKAN then LeanKAN]
        self.kin = nn.Sequential(
            AddKANLayer(m + 1, hidden, num_basis),
            LeanKANLayer(hidden, m, n_mu=n_mu, num_basis=num_basis),
        )
        self.use_thermo = use_thermo
        if use_thermo:
            self.thermo_linear = nn.Linear(m, 1, bias=False)   # Eq. 14 linear part
            self.cor = AddKANLayer(m + 1, 1, num_basis)        # Eq. 17 KAN_cor

    def forward(self, t, u):                # (B, m+1) if thermo else (B, m)
        if not self.use_thermo:
            return self.kin(u)              # biodiesel: species rates only
        dY = self.kin(u)                    # (B, m)  species rates
        dT = self.thermo_linear(dY) + self.cor(u)   # (B, 1)  Eq. 15
        return torch.cat([dY, dT], dim=-1)  # (B, m+1)
```

### Check

- **Biodiesel** (`use_thermo=False`, input/output = 6 species): drop this into the
  Milestone-4 training loop, now on *all* 20 trajectories. It should fit the clean
  data and — the paper's headline biodiesel result — **not overfit** when you train
  on the 5/10/15 % noisy arrays (`train_states_noise05/10/15`).
- **H₂** (`use_thermo=True`, `m=9`, input/output = 10): just instantiate it and
  confirm a forward pass returns shape `(B, 10)`. Training it is Milestone 6–7.

---

## Milestone 6 — two-stage training + the loss (Eq. 18)

**Goal:** train the way the paper does — kinetics first, thermodynamics second.

### Why two stages

Learning species rates and the temperature rate at once is hard (they're coupled
and stiff). The paper splits it:

- **Stage 1 — kinetics.** Train only `θ_kin`. Loss on the `m` species profiles
  (`n* = m` in Eq. 18). Temperature is just *read from the data* as an input. For
  isothermal biodiesel, **Stage 1 is the whole model.**
- **Stage 2 — thermodynamics.** Freeze/stabilize the converged core, add the
  thermo superstructure, and learn `dT/dt`. Loss on all `m+1` profiles (`n* = m+1`).

### The loss (Eq. 18)

$$\mathcal L(\boldsymbol\theta) = \mathcal L_{\text{MSE}}(\boldsymbol\theta) + \mathcal L_{\text{PINN}}(\boldsymbol\theta)$$

$$= \underbrace{\frac{1}{n^*}\sum_{j=1}^{N_t}\sum_{k=1}^{n^*}\big(\hat u^{\text{pred}}_k(t_j,\boldsymbol\theta) - \hat u^{\text{obs}}_k(t_j)\big)^2}_{\text{MSE, variable } n^*} \; + \; \alpha_{\text{PINN}}\,\underbrace{\sum_{i=1}^{N_e}\sum_{j=1}^{N_t}\bigg|\sum_{k=1}^{m} \frac{N_i^k\, W_i\, (Y^{\text{pred}}_{k,j}-Y^{\text{pred}}_{k,1})}{W_k}\bigg|}_{\text{optional element conservation}}$$

where `n* = m` in Stage 1 and `n* = m+1` in Stage 2.

- **MSE** on the `[0,1]`-normalized states (min–max, **train-only** — reuse your
  `u_min/u_max`), summed over the `N_t` time points `t_j` and the `n*`
  thermochemical states. This is the main term.
- **PINN term** (optional): penalizes drift in the `N_e` conserved elements (H, O,
  N) across all `N_t` timesteps — the inner `Σ_{k=1}^{m}` is the element-mass
  difference from the initial condition, with `N_i^k` = atom count of element `i`
  in species `k`, `W_i` its atomic mass, `W_k` the species molar mass.
  `α_PINN = 1e-4`. The paper uses **MSE only for biodiesel**, and **MSE + PINN for
  H₂**. Add it *after* plain MSE works.
- Optimizer: **Adam, lr = 2e-3**.

### Code you write (training skeleton)

```python
def mse(pred, obs):                       # both (T, B, dim) normalized
    return ((pred - obs) ** 2).mean()

def train_stage(func, u0, t, target, steps, params, lr=2e-3):
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        opt.zero_grad()
        pred = odeint(func, u0, t, method="dopri5")     # (T, B, dim)
        loss = mse(pred, target)
        loss.backward(); opt.step()
        if step % 100 == 0: print(step, loss.item())
    return func

# Stage 1: kinetics only  (biodiesel finishes here)
model = ChemKAN(m=..., use_thermo=False)
train_stage(model, u0, t, target_species, steps=..., params=model.kin.parameters())

# Stage 2 (H2 only): switch on thermo, train the superstructure
model.use_thermo = True
thermo_params = list(model.thermo_linear.parameters()) + list(model.cor.parameters())
train_stage(model, u0, t, target_full, steps=..., params=thermo_params)
```

### Check

- **Biodiesel:** reproduce the paper's robustness claim — train separately on the
  0/5/10/15 % noise arrays; test error should stay low and *not* diverge from
  train error (no overfitting).
- **H₂:** after Stage 2, the temperature profile ignites at the right time. Compare
  your predicted ignition delay to the `ignition_delay` array already stored in
  `hydrogen.npz` by the data generator.

---

## Milestone 7 — validate against the paper

- **Parameter count.** Print `sum(p.numel() for p in model.parameters())`. The
  paper's H₂ ChemKAN is **344 parameters**. You won't hit it exactly (it depends on
  `hidden`, `num_basis`, `n_mu`), but you should be in the same order — if you're at
  10k, your layers are too wide; shrink `hidden`/`num_basis`. Being *lean* is the point.
- **Biodiesel:** trajectories match under noise, no overfitting (paper's Fig. for the
  biodiesel case).
- **H₂:** correct ignition timing and a ~2× speed-up vs. the detailed Cantera solve
  for the same trajectory (the paper's acceleration claim). Measure wall-clock of a
  `ChemKAN` rollout vs. a Cantera integration over the same window.
- Cross-check activations/loss against **DENG-MIT/KAN-ODEs** and **LeanKAN** if a
  number looks off.

### Suggested `src/` layout (build it as you go)

```
chemkan/src/
  addkan/addkan.py          # (existing) B-spline reference — keep for comparison
  kan/
    rbf.py                  # RBFActivation                 (Milestone 1)
    layers.py               # AddKANLayer, LeanKANLayer      (Milestone 2-3)
    kanode.py               # KANODE + rollout              (Milestone 4)
  chemkan/
    model.py                # ChemKAN (kinetic core + thermo superstructure)  (M5)
    train.py                # two-stage training + Eq. 18 loss                (M6)
    losses.py               # mse + optional PINN element conservation
tests/
  test_layers.py            # n_mu=0 == AddKAN; shapes; product-fit sanity
```

---

## Cheat-sheet: equation → code

| Paper | Concept | Where you build it |
|---|---|---|
| Eq. 11–12 | RBF activation `φ_{l,α,β}(x)` | `RBFActivation` (M1) |
| Eq. 7 (node op = Eq. 10, `n_l^{mu}=0`) | AddKAN activation matrix `Φ_l` | `AddKANLayer` (M2) |
| Eq. 8–10 | LeanKAN layer (`y_l^{mult} + y_l^{add}`) | `LeanKANLayer` (M3) |
| Eq. 6 | KAN as ODE right-hand side | `KANODE` + `odeint` (M4) |
| Eq. 13, 16 | Kinetic core `KAN_kin` | `ChemKAN.kin` (M5) |
| Eq. 14–15, 17 | Thermo superstructure | `ChemKAN.thermo_linear` + `.cor` (M5) |
| Eq. 18 | Loss (MSE + optional PINN) | `train.py` / `losses.py` (M6) |

## Common beginner pitfalls

- **Forgetting to normalize** states (and time). Stiff ODEs blow up; the `[0,1]`
  scaling is not optional. Reuse the *train-only* `u_min/u_max` from your `.npz`.
- **Empty-product bug** in LeanKAN (`n_mu=0` ⇒ `prod` returns 1). Guard it.
- **tanh saturation.** If inputs are far outside `[-1,1]`, `tanh` flattens and
  gradients vanish. Normalize inputs before the first layer.
- **Using the B-spline `addkan.py` as-is.** It's a structural reference; ChemKAN's
  basis is RBF (Eq. 11–12). Don't mix them up.
- **Jumping to H₂ first.** It's the stiff one. Get biodiesel (isothermal,
  kinetics-only) working end-to-end before adding thermo and stiffness together.
- **Wrong solver expectation.** `torchdiffeq` is non-stiff by default; if H₂ won't
  converge, that's expected — normalize time, shrink the window, and read the
  paper's forward-sensitivity note.
```
