"""`seedmesh setup` -- get a working backend onto a volunteer's machine in one command.

Petals has been unmaintained since 2024-09-07 and does not run on current dependencies
without patching (see docs/petals-port.md). The patches exist as a codemod, but expecting a
volunteer to clone a repo, run a codemod against it and pip-install the result is not an
onboarding story -- it is a support conversation with every single person.

So this automates exactly that: clone, patch, install, verify. The codemod is idempotent,
so re-running is safe, and every step prints what it did rather than hiding it.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

PETALS_REPO = "https://github.com/bigscience-workshop/petals.git"
DEFAULT_DIR = Path.home() / ".seedmesh" / "petals"

# Runtime dependencies Petals needs that its own metadata pins to unusable versions, plus
# ones a --no-deps install skips. cpufeature is not optional on x86_64: lm_head.py guards
# its import on platform.machine(), not on the package being present.
RUNTIME_DEPS = [
    "torch",
    "transformers>=4.48",
    "accelerate",
    "bitsandbytes",
    "hivemind==1.1.12",
    "dijkstar",
    "humanfriendly",
    "async-timeout",
    "sentencepiece",
    "peft",
    "speedtest-cli",
    "requests",
    "tensor_parallel==1.0.23",
]


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(command[:6])}{' ...' if len(command) > 6 else ''}")
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"command failed with exit {result.returncode}")
    return result


def check_platform() -> list[str]:
    """Report anything that will stop this working, before doing any work."""
    problems = []
    if platform.system() == "Windows":
        problems.append(
            "hivemind does not run natively on Windows. Use WSL2:\n"
            "      wsl --install -d Ubuntu, then run this inside the Ubuntu shell.\n"
            "      (Seedmesh's own trust layer runs fine on Windows; the backend does not.)"
        )
    if sys.version_info < (3, 9):
        problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old; need >= 3.9")
    return problems


def cmd_setup(args) -> int:
    print("seedmesh setup -- installing and patching the Petals backend\n")

    problems = check_platform()
    if problems and not args.force:
        print("cannot continue:")
        for problem in problems:
            print(f"    {problem}")
        print("\n  (--force to try anyway)")
        return 2

    target = Path(args.petals_dir).expanduser()
    print(f"[1/4] Petals checkout at {target}")
    if (target / ".git").exists():
        print("  already present; leaving as-is (delete it to start clean)")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", PETALS_REPO, str(target)])

    print("\n[2/4] applying the Seedmesh port")
    port_script = Path(__file__).resolve().parents[2] / "tools" / "port_petals.py"
    if not port_script.exists():
        print(f"  cannot find {port_script}; run this from a Seedmesh checkout")
        return 2
    result = run([sys.executable, str(port_script), "--petals-root", str(target)], check=False)
    print("    " + "\n    ".join(result.stdout.strip().splitlines()[-8:]))
    if result.returncode != 0:
        print("  port failed -- see output above")
        return 1

    if args.skip_install:
        print("\n[3/4] skipping dependency install (--skip-install)")
    else:
        print("\n[3/4] installing dependencies (this is the slow part)")
        run([sys.executable, "-m", "pip", "install", "--quiet", *RUNTIME_DEPS])
        print("  installing the patched Petals (no deps -- its pins are the thing we patched)")
        run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(target)])

    print("\n[4/4] verifying")
    verify = Path(__file__).resolve().parents[2] / "tools" / "verify_petals_port.py"
    if verify.exists():
        result = run([sys.executable, str(verify)], check=False)
        tail = result.stdout.strip().splitlines()[-10:]
        print("    " + "\n    ".join(tail))
        if result.returncode != 0:
            print("\n  verification FAILED -- the backend is not usable yet")
            return 1
    print("\nready. Next:")
    print("  seedmesh probe --model <model>     # how many blocks can I host?")
    print("  seedmesh serve --model <model> --initial-peers <addr>")
    return 0
