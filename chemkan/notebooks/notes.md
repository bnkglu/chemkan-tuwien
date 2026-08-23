# Notes — implementing KAN from scratch (recap)

Working companion to `01_implement_kan.ipynb` and
`docs/implementing_chemkan_from_scratch.md`. Covers what we built and the mechanics
behind it (indices, shapes, broadcasting, einsum).

---

## Big picture

Building ChemKAN in milestones: **RBF activation → AddKAN layer → LeanKAN layer →
KAN-ODE → ChemKAN model → training**. We use **Gaussian RBF** activations (paper
Eq. 11-12), *not* B-splines. Done so far: Milestone 1 (activation) and Milestone 2
(AddKAN layer).

**The one idea.** An MLP edge has a fixed activation and a learnable *scalar*:
`y_i = σ(Σ_j w_ij x_j)`. A KAN edge has a learnable *function*:
`y_i = Σ_j φ_ij(x_j)`. → **learnable activation functions on the edges, sum at the nodes.**

---

## Milestone 1 — one activation `φ(x)` (Eq. 11-12)

$$\phi(x)=\sum_{i=1}^{N} w^{\psi}_i\,\exp\!\Big(\tfrac{-(x-c_i)^2}{2h^2}\Big)+w^{b}\,b(x)$$

A curve made of `N` Gaussian **bumps**; training just picks their heights.

| symbol | code | role | learned? |
|---|---|---|---|
| `N` | `num_basis` | how many bumps = curve resolution | no |
| `c_i` | `centers` (buffer) | bump positions (a uniform grid) | no |
| `h` | `self.h` | bump width = grid spacing | no |
| `w^ψ_i` | `w_rbf` | bump **heights** — the shape of φ | **yes** |
| `w^b` | `w_base` | weight on the SiLU residual path | **yes** |
| `b(x)` | `nn.SiLU()` | base activation (a sensible nonzero start) | fixed |

```python
def forward(self, x):                       # x: (batch,)
    r   = x.unsqueeze(-1) - self.centers    # (batch, num_basis)   distance to each center
    psi = torch.exp(-r**2 / (2*self.h**2))  # (batch, num_basis)   bump values
    return psi @ self.w_rbf + self.w_base * self.base(x)   # (batch,)
```

Intuition confirmed by the "watch it learn" cell: bumps start flat, stretch to
different heights (some negative), and their **sum** traces `sin(3x)`.

---

## Tensor mechanics we learned (these carry through every milestone)

**Batch = the input already is the batch.** `x` of shape `(batch,)` is many inputs
at once. You never loop or "extract" it — every op is vectorized over the batch.
(In M1 there are **no features** yet: one scalar per sample. Features arrive in M2.)

**Adding an axis for broadcasting.** To compute `x[b] - c_i` for every pair we need
shapes that broadcast:
- `x.unsqueeze(-1)` / `x[..., None]`  → **adds** a trailing size-1 axis: `(batch,) → (batch,1)`.
- `flatten()` does the **opposite** (removes axes) — wrong tool here.
- bare `x[:]` returns `x` unchanged (no new axis) → `(batch,) - (num_basis,)` **errors**.
- Use `x.unsqueeze(-1)` (not `x[:,None]`): it means "append an axis" and works in
  M1 *and* M2 (where `x` is 2-D). **Don't reassign `x`** — keep the original for `base(x)`.

Broadcasting rule: a size-**1** axis stretches to match. `(batch,1) - (num_basis,)`
→ `(batch, num_basis)`; `r[b,i] = x[b] - c_i`.

**The numerator `-r**2`.** `r` already *is* `x - c_i`, so `-r**2 = -(x-c_i)^2`. The
paper's norm `‖x-c_i‖` disappears once squared (sign gone) — no `abs()` needed.
⚠️ Precedence: write `-r**2 / (2*self.h**2)` **with parentheses**. `-r**2 / 2 * h**2`
means `-r²·h²/2` (multiplies by h² instead of dividing) → flat bumps, no fit.

**`psi @ w_rbf` vs paper's `w·ψ`.** The paper's `Σ w_i ψ_i` is *scalar* multiply,
which commutes, so order is just convention. `@` is *matmul*: order is forced by
shapes — it contracts the **last axis of the left** with the axis of the right.
`psi (b,N) @ w_rbf (N,) → (b,)` sums over `N`. Same as `(w_rbf * psi).sum(-1)`
(that version reads in the paper's order). Reversed `w_rbf @ psi` → shape error.

**Superscripts are labels, not powers.** `w^ψ` = "weight on the ψ term",
`w^b` = "weight on the base b(x)". Nothing is raised to a power.

---

## Milestone 2 — AddKAN layer (Eq. 7; node op = Eq. 10, n_mu=0)

Eq. 7 is the **matrix of activations** `Φ_l` (size `out × in`). The node operation
(how output `i` is formed) is the additive case:

$$y_{l,i}=\sum_{j=1}^{n_l}\phi_{l,i,j}(x_{l,j})$$

One learnable φ per **(output i, input j)** edge → `out × in` activations.

**Shapes (memorize):**
```
x:(B,in) → tanh → r:(B,in,N) → psi:(B,in,N)
w_rbf:(out,in,N)   w_base:(out,in)          → y:(B,out)
```

**`self.centers` is created in `__init__`** (`register_buffer`), just like M1. There
is **one shared grid** `(N,)` for all edges — centers/`h` fix *where/how wide* the
bumps are; only `w_rbf` (per edge) sets their heights.

**Why `tanh(x)` first.** The RBF grid lives on `[-1,1]`. `tanh` squashes any real
input into that range so it lands on the grid; otherwise off-grid inputs make every
Gaussian ≈ 0 and the layer goes deaf. (Paper's alternative to re-gridding.)

**Why init `* 0.1`.** `randn` has std 1. Each output **sums over `in × N` terms**;
summed random terms have variance that grows with the count, so std-1 init gives
large, unstable outputs (saturates the next `tanh`). Scaling to std 0.1 keeps
initial outputs small and training stable. `0.1` is a hand-picked small scale (same
spirit as Xavier/Kaiming fan-in scaling).

**Why `w_base` has no `N` axis.** The base term is a **single** function `w^b·b(x)`,
not a sum over bumps — no `i` index in Eq. 11's base term → no `num_basis` axis. So
per edge: `N` bump heights **+ 1** base weight → `w_rbf:(out,in,N)`, `w_base:(out,in)`.

### einsum (the layer's two contractions)

Rules: (1) letter repeated across inputs → multiply; (2) letter absent from output
→ **sum it away**; (3) letter in output → keep. Letters: `b`=batch, `i`=in,
`k`=num_basis, `o`=out.

```python
rbf  = torch.einsum("bik,oik->bo", psi, self.w_rbf)          # sum over i AND k
base = torch.einsum("bi,oi->bo",  self.base(x), self.w_base) # sum over i
```

- `"bik,oik->bo"`: `psi(b,i,k) * w_rbf(o,i,k)`, sum over `i` and `k`, keep `b,o`.
  → `out[b,o] = Σ_i Σ_k psi[b,i,k]·w_rbf[o,i,k]`. Sums over `k` build each edge's φ;
  sum over `i` is the node addition — both at once.
- `"bi,oi->bo"`: base term, only `i` to sum (no bumps). Identical to `base_x @ w_base.T`
  = `F.linear(base_x, w_base)` — an ordinary linear layer.

Pattern: **keep `b` and `o`, sum over the input-side axes.**

---

## Training loop (M1 check), line by line

```python
torch.manual_seed(0)                 # reproducible random init
phi = RBFActivation(num_basis=12)    # 12 bumps
opt = torch.optim.Adam(phi.parameters(), lr=0.05)   # updates w_rbf, w_base
x = torch.linspace(-1,1,200); y = torch.sin(3*x)    # full-batch dataset
for _ in range(2000):                # 2000 full-batch "epochs"
    opt.zero_grad()                  # clear .grad — backward ACCUMULATES (adds), so
                                     #   old grads must be wiped or they pile up
    loss = ((phi(x) - y)**2).mean()  # MSE: prediction vs target
    loss.backward()                  # autograd fills param.grad = d(loss)/d(param)
    opt.step()                       # Adam moves params along .grad
```
Order is always **zero_grad → forward/loss → backward → step**.

**`.detach()` when plotting.** `phi(x)` still carries the autograd graph; converting
it to numpy for matplotlib errors. `.detach()` returns a graph-free copy so it can
be drawn. Black = target, red = learned φ (a visual comparison). **bumps** =
`w_rbf[i] · gaussian_i` per basis function; the grey curves sum (≈) to red φ.

---

## Gotchas we hit (and for later)

- **Operator precedence** in the Gaussian → parenthesize `/(2*self.h**2)`.
- **Overwriting `x = x[:,None]`** made `base(x)` shape `(B,1)`, broadcasting the
  output to `(B,B)` (hence the 200-line plot). Keep `x` as `(batch,)`.
- **Empty-product trap (LeanKAN, M3):** `g[..., :0].prod(-1) == 1`, not 0 — guard `n_mu==0`.
- **Always `print(out.shape)`** when unsure; a surprise extra axis is usually a
  broadcasting bug.

---

## Next

M3 LeanKAN (add multiplication, `n_mu`), M4 KAN-ODE (predict the rate, integrate),
then the ChemKAN model + two-stage training. Biodiesel (isothermal, non-stiff)
before hydrogen (stiff). Run `pytest chemkan/tests/test_layers.py -v` as you promote
each class into `src/`.
