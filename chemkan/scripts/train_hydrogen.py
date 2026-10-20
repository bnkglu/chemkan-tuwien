r"""Train the hydrogen ChemKAN (paper Sec. III B), one stage per call.

Stage 1 trains the kinetic core with the observed temperature; Stage 2 starts from a Stage-1
archive and trains the full [Y, T] model. The implementation, shared with the other
combustion system, is ``combustion/train_combustion.py`` (its docstring lists the model, the
dynamics and the loss). Every option is shown by ``--help`` of each stage:

    python chemkan/scripts/train_hydrogen.py stage1 --help
    python chemkan/scripts/train_hydrogen.py stage2 --help

The main run (results/hydrogen/):

    python chemkan/scripts/train_hydrogen.py stage1 --run-dir runs/hydrogen/S1_seed0 --epochs 100000
    python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/hydrogen/S2_seed0 \
        --stage1-archive runs/hydrogen/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt --epochs 100000

An interrupted or extended stage continues with ``--resume`` (and a larger ``--epochs``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "combustion"))
import train_combustion  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(train_combustion.main(system="hydrogen"))
