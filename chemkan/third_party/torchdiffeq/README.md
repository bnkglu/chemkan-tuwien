# torchdiffeq (third-party, MIT license)

Source distribution of the torchdiffeq commit that ChemKAN depends on, kept so that the
environment can be rebuilt without GitHub.

- **Upstream:** https://github.com/rtqichen/torchdiffeq
- **Commit:** `657943acefa826ef04c025ebeb1ff5e9d60dc268` (2025-04-03, merge of PR #265); the
  package reports version 0.2.5. This is the commit pinned in `../../requirements.txt` and
  `../../pyproject.toml`. It provides `method="tsit5"`, which the PyPI release does not.
- **File:** `torchdiffeq-0.2.5.tar.gz`, built from that commit with `python -m build --sdist`.
  sha256 `35ae6e19c695a6174ebc54eace5e0bfd812982815a9f24fc6a271ec665d4fb7a`.
- **License:** `LICENSE` (MIT, Copyright (c) 2018 Ricky Tian Qi Chen), copied unchanged from
  the commit; it is also inside the archive.

Installing the archive gives the same `torchdiffeq/` files as the project venv installed from
GitHub (checked with `diff -r`, 2026-10-20).

Normally `pip install -r requirements.txt` fetches this commit from GitHub. If that fails
(GitHub or the commit unavailable), install the archive instead, from `chemkan/`:

    pip install third_party/torchdiffeq/torchdiffeq-0.2.5.tar.gz

then remove the `torchdiffeq @ git+...` line from `requirements.txt` and run
`pip install -r requirements.txt` for the other packages.
