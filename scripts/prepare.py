"""Download Metrica's public sample data (if needed) and cache the 3 matches.

    python scripts/prepare.py [--sample-dir PATH]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample-dir", type=Path, default=data.ROOT / "data" / "metrica-sample-data")
args = ap.parse_args()

if not args.sample_dir.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/metrica-sports/sample-data.git", str(args.sample_dir)],
        check=True,
    )
for n in data.prepare_samples(args.sample_dir):
    m = data.load(n)
    print(f"{n}: {m.n_frames} frames @ {m.fps:.0f} fps, {m.home.shape[1]}+{m.away.shape[1]} players, {len(m.events)} events")
