# Hydrogen Stage-1 temperature resolution

Saved console output of
[`chemkan/scripts/benchmark/benchmark_temperature_resolution.py`](../../../chemkan/scripts/benchmark/benchmark_temperature_resolution.py)
(moved here from `results/benchmark_temperature_resolution.txt`; content unchanged).

Stage 1 drives the hydrogen kinetic core with a temperature profile interpolated from a dense
precomputed Cantera trajectory. The benchmark measures how far that interpolated temperature
is from Cantera evaluated directly, for several cache resolutions and for linear vs PCHIP
interpolation, together with generation time, interpolation time and cache size. It is the
measurement behind the 20,000-point cache (`chemkan/data/generated/hydrogen_temperature_20000.npz`).

`benchmark_temperature_resolution.txt` holds four consecutive runs (four `SUMMARY` tables),
all on the 35 training conditions, 0–0.6 ms, evaluated at the 839 recorded Tsit5
right-hand-side query times of one Stage-1 integration with the trained checkpoint
(`--times solver`, the notebook 05 §9.5b times), rtol 1e-6 / atol 1e-8:

| run | resolutions (points) |
|---|---|
| 1 | 2,000 · 20,000 · 100,000 · 200,000 · 10,000,000 |
| 2 | 30,000 · 20,000 |
| 3 | 20,000 – 65,000 |
| 4 | 20,000 – 100,000 |

Reproduce (from the repository root):

```bash
python chemkan/scripts/benchmark/benchmark_temperature_resolution.py \
    --times solver --resolutions 2000 20000 200000
```

Related: [`chemkan/notebooks/05_temperature_flow_audit.ipynb`](../../../chemkan/notebooks/05_temperature_flow_audit.ipynb)
(§9 interpolation alternatives, §10–11 effect on Stage-1 training).
