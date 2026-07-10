# ChemKAN reproduction

An attempted reproduction of the datasets and (later) models from Koenig, Kim &
Deng (2025), *ChemKANs for combustion chemistry modeling and acceleration*.

**Current status:** the data-generation scripts are implemented under
`scripts/data_gen`. Model / training code will be added later under `src`.

## Layout

```
chemkan/
├── README.md                # this file
├── requirements.txt
├── data/
│   └── generated/           # generated .npz files land here (not committed)
├── scripts/
│   └── data_gen/            # data-generation scripts + their README
└── src/                     # reserved for later model / training code
```

## Data generation

Biodiesel and hydrogen are the reproduction datasets; methane is an optional
extension. See **[`scripts/data_gen/README.md`](scripts/data_gen/README.md)**
for setup, quickstart commands, generated files, verification, and
implementation notes.

## Reference

Koenig, B. C., Kim, S., & Deng, S. (2025). ChemKANs for combustion chemistry
modeling and acceleration. *Physical Chemistry Chemical Physics*, 27,
17313–17330.
