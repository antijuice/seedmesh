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

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def has_nvidia_gpu() -> bool:
    import shutil

    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(command[:6])}{' ...' if len(command) > 6 else ''}")
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"command failed with exit {result.returncode}")
    return result


def has_c_compiler() -> bool:
    """Whether Triton will be able to build its CUDA kernels.

    Checks `CC` first because Triton does; then the usual names. `cc` alone is not enough on
    every distro, so any of them counts.
    """
    import os
    import shutil

    explicit = os.environ.get("CC")
    if explicit and shutil.which(explicit):
        return True
    return any(shutil.which(name) for name in ("cc", "gcc", "clang"))


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

    # A GPU means torch will route some ops through Triton, which JIT-COMPILES CUDA kernels
    # at runtime and shells out to a C compiler to do it. Without one, the failure arrives
    # ~60 frames deep on the first forward pass, inside throughput measurement, reading
    # "Failed to find C compiler" with no hint that it is a missing apt package. Reported by
    # a volunteer whose machine was otherwise correctly set up.
    if has_nvidia_gpu() and not has_c_compiler():
        problems.append(
            "no C compiler found, and this machine has an NVIDIA GPU.\n"
            "      torch compiles Triton kernels at runtime and needs one, so serving would\n"
            "      fail on its first forward pass. Install one:\n"
            "        sudo apt install -y build-essential            # Debian/Ubuntu\n"
            "        sudo dnf groupinstall -y 'Development Tools'   # Fedora/RHEL\n"
            "      (or set CC=/path/to/compiler if you have one elsewhere)"
        )
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

    # Dependencies BEFORE the port: the codemod verifies its symbol mapping against the
    # *installed* hivemind before rewriting anything, so it cannot run until hivemind
    # exists. (An earlier version patched first and failed with a wall of
    # "No module named 'hivemind'".)
    if args.skip_install:
        print("\n[2/4] skipping dependency install (--skip-install)")
    else:
        cpu_only = args.cpu_torch or not has_nvidia_gpu()
        print(f"\n[2/4] installing dependencies (the slow part; "
              f"{'CPU-only torch' if cpu_only else 'CUDA torch'})")
        if cpu_only:
            # A bootstrap peer hosts no blocks and a CUDA torch is ~2.5 GiB of wheels it
            # will never use -- enough to OOM a 1 GiB VPS during install.
            print("  no NVIDIA GPU detected; using CPU torch wheels (--cpu-torch to force)")
            run([sys.executable, "-m", "pip", "install", "--quiet",
                 "torch", "--index-url", CPU_TORCH_INDEX])
        else:
            run([sys.executable, "-m", "pip", "install", "--quiet", "torch"])
        run([sys.executable, "-m", "pip", "install", "--quiet", *RUNTIME_DEPS])

    print("\n[3/4] applying the Seedmesh port")
    port_script = Path(__file__).resolve().parents[2] / "tools" / "port_petals.py"
    if not port_script.exists():
        print(f"  cannot find {port_script}; run this from a Seedmesh checkout")
        return 2
    result = run([sys.executable, str(port_script), "--petals-root", str(target)], check=False)
    print("    " + "\n    ".join(result.stdout.strip().splitlines()[-9:]))
    if result.returncode != 0:
        print("  port failed -- see output above")
        return 1

    if not args.skip_install:
        print("  installing the patched Petals (no deps -- its pins are what we patched)")
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
