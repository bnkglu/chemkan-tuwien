r"""Pull iCloud-evicted repository files back to local disk.

macOS "Optimize Mac Storage" evicts files in an iCloud-synced folder to placeholders.
A read then triggers an asynchronous fetch, and a read attempted before the fetch
completes fails -- as EOFError for a data file, or as TimeoutError partway through an
import for a source file. Run this before a long job so every file the run touches is
resident on disk first.

    python3 chemkan/scripts/diagnostics/materialize_icloud.py <repo-root>
"""

import os, subprocess, sys, time
from pathlib import Path
ROOT = Path(sys.argv[1])
EXT = {".npz", ".pt", ".json", ".csv", ".ipynb", ".py", ".yaml", ".md"}
targets = [p for d in ("chemkan/data", "chemkan/notebooks", "chemkan/scripts",
          "chemkan/src", "chemkan/tests", "results")
           for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in EXT]
def dataless(p):
    try:
        with open(p, "rb") as f:
            return len(f.read(1)) == 0 and p.stat().st_size > 0
    except OSError:
        return True
pending = [p for p in targets if dataless(p)]
print(f"{len(targets)} candidate files, {len(pending)} evicted")
for _ in range(40):
    if not pending:
        break
    for p in pending:
        subprocess.run(["/usr/bin/brctl", "download", str(p)], capture_output=True)
    time.sleep(3)
    pending = [p for p in pending if dataless(p)]
    print(f"  still evicted: {len(pending)}")
if pending:
    print("FAILED to materialize:"); [print("   ", p.relative_to(ROOT)) for p in pending[:20]]
    sys.exit(1)
print("all files materialized")
