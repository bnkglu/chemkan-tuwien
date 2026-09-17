"""Run a training script in-process with resume-test hooks (validation tool only).

    python fsa/_resume_driver.py [--interrupt-stage S --interrupt-epoch N] SCRIPT ARGS...

* ``--interrupt-epoch N`` simulates Ctrl+C right after the script saves its resume
  checkpoint for epoch ``N`` (of stage ``S`` for hydrogen), so the script's own
  KeyboardInterrupt path runs and ``checkpoint_resume.pt`` survives.
* Every resume checkpoint the script writes is also copied to
  ``resume_history/<stage>_<epoch>.pt`` (the script itself overwrites and finally deletes
  ``checkpoint_resume.pt``), so optimizer and RNG states can be compared afterwards.

The training code itself is not modified; only ``RunManager`` methods are wrapped.
"""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import _run  # noqa: E402


def main():
    argv = sys.argv[1:]
    stage, epoch = None, None
    while argv and argv[0].startswith("--interrupt"):
        flag, value = argv[0], argv[1]
        argv = argv[2:]
        if flag == "--interrupt-stage":
            stage = value
        elif flag == "--interrupt-epoch":
            epoch = int(value)
    script, *args = argv

    save_resume = _run.RunManager.save_resume

    def save_resume_hook(self, state):
        save_resume(self, state)
        keep = self.run_dir / "resume_history"
        keep.mkdir(exist_ok=True)
        shutil.copy(self.resume_path, keep / f"{state.get('stage')}_{state.get('epoch')}.pt")
        if epoch is not None and state.get("epoch") == epoch and \
                (stage is None or state.get("stage") == stage):
            raise KeyboardInterrupt                     # right after the checkpoint is on disk

    _run.RunManager.save_resume = save_resume_hook
    sys.argv = [str(SCRIPTS / script), *args]
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")


if __name__ == "__main__":
    main()
