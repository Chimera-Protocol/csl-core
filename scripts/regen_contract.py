#!/usr/bin/env python3
"""
Regenerate the 0.5.1 compatibility goldens under tests/contract/golden/.

    python scripts/regen_contract.py            # from tag v0.5.1 (the only intended use)
    python scripts/regen_contract.py --tag v0.5.1 --check   # regenerate into a temp dir and diff

The tagged source is exported with `git archive` into a temporary folder and captured
in a fresh interpreter, so the goldens reflect the tag and never the working tree.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "tests" / "contract"
sys.path.insert(0, str(CONTRACT))
import capture  # noqa: E402


def export_tag(tag: str, dest: Path) -> str:
    commit = subprocess.check_output(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=REPO, text=True).strip()
    archive = dest.parent / "src.tar"
    with open(archive, "wb") as f:
        subprocess.check_call(["git", "archive", "--format=tar", commit], cwd=REPO, stdout=f)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as t:
        t.extractall(dest, filter="data")
    return commit


def run_capture(src: Path, golden: Path, extra: list[str]) -> None:
    cmd = [sys.executable, str(CONTRACT / "capture.py"), "--src", str(src), "--golden", str(golden), *extra]
    subprocess.check_call(cmd, cwd=src)


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def build(tag: str, out: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        commit = export_tag(tag, src)

        if out.exists():
            shutil.rmtree(out)
        (out / "policies").mkdir(parents=True)
        for key, path in capture.corpus(src).items():
            shutil.copyfile(path, out / "policies" / f"{key}.csl")
        # Probe policies written for the contract: they exercise every operator the
        # examples leave out. Their goldens still come from the tagged engine.
        for path in sorted((CONTRACT / "probes").glob("*.csl")):
            shutil.copyfile(path, out / "policies" / f"contract__probes__{path.stem}.csl")

        run_capture(src, out, ["--make-inputs"])
        data = capture.run_sharded(src, out)

    for rel, content in capture.split_capture(data).items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    manifest = {
        "tag": tag,
        "commit": commit,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "z3_solver": _ver("z3-solver"),
        "rich": _ver("rich"),
        "mcp": _ver("mcp"),
        "normalisation": capture.NORMALISATION,
        "policies": len(list((out / "policies").glob("*.csl"))),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v0.5.1")
    ap.add_argument("--out", default=str(CONTRACT / "golden"))
    ap.add_argument("--check", action="store_true", help="regenerate into a temp dir and diff against --out")
    args = ap.parse_args()

    if not args.check:
        build(args.tag, Path(args.out))
        print(f"goldens written to {args.out}")
        return 0

    with tempfile.TemporaryDirectory() as td:
        fresh = Path(td) / "golden"
        build(args.tag, fresh)
        rc = subprocess.call(
            ["diff", "-r", "-q", "-x", "manifest.json", str(fresh), args.out]
        )
    print("goldens reproducible" if rc == 0 else "goldens differ from a fresh regeneration")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
