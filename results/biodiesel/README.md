# Author-matched biodiesel experiment

Runs, Julia reference exports and provenance for the match to the authors' released
biodiesel example. What each run is and what it shows: [`docs/authors_materials.md`](../../../../docs/authors_materials.md).

## Figure runs (`figures/`)

`chemkan/scripts/author_repo_match/run_figures.py` trains the Fig. 4 and Fig. 5 runs on
`chemkan/data/generated/biodiesel_v2.npz`, seed 0 (Figs. 3 and 6 are evaluated from the
Fig. 5 runs; sizes and their sources: `docs/authors_materials.md` §3):

| group | model | settings | epochs |
|---|---|---|---|
| `fig05/` | ChemKAN | `train_author_repo_match.py --data canonical --hidden 4 --solve-mode batched --sensitivity fsa`, noise 0, 1, 2, 3, 5, 7, 10, 15 % | 10,000 |
| `fig05/` | DeepONet 308 | `train_biodiesel_deeponet.py --width 8 --eval-every 1`, same noise levels | 10,000 |
| `fig04/` | ChemKAN | `--hidden 2, 3, 4, 9, 11`, same settings, noise-free | 5,000 |
| `fig04/` | DeepONet | `FIG4_ARCHITECTURES` 78, 156, 249, 308, 384, 456 (`--width w --trunk-hidden q`), noise-free | 50,000 |

Run from the repository root:

```bash
P=~/uni_projects/chemkan-venv/bin/python
$P chemkan/scripts/author_repo_match/run_figures.py --list            # runs, status, time estimates
$P chemkan/scripts/author_repo_match/run_figures.py --dry-run          # commands only
$P chemkan/scripts/author_repo_match/run_figures.py                   # everything, one at a time
$P chemkan/scripts/author_repo_match/run_figures.py --only fig05_chemkan_noise05
$P chemkan/scripts/author_repo_match/run_figures.py --shard 1/3       # one of 3 time-balanced groups
$P chemkan/scripts/author_repo_match/run_figures.py --jobs 2          # 2 in parallel, 1 thread each
```

Finished runs (`checkpoint_final.pt`) are skipped and interrupted ones resume from
`checkpoint_resume.pt`, so the same command can simply be rerun. The shard split is computed
over all 27 runs, so every machine gets the same groups. Starts and finishes are logged to
`figures/run_log_<hostname>.txt`; each run's console output goes to `<run>.console.log`.
