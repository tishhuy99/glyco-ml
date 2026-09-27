"""Show what the project is using on disk, and optionally delete regenerables.

    python src/cleanup.py           # report only
    python src/cleanup.py --purge   # delete feature matrices (regenerable)
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true",
                    help="delete X_*.npy and rowmask_*.npy (regenerate with featurize.py)")
    args = ap.parse_args()

    rows = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            rows.append((p.stat().st_size, p.relative_to(ROOT)))
    rows.sort(reverse=True)

    print(f"total project: {size(ROOT) / 1e6:,.0f} MB\n")
    print("files over 1 MB:")
    for sz, rel in rows:
        print(f"  {sz / 1e6:>8,.0f} MB  {rel}")

    regen = list((ROOT / "data" / "processed").glob("X_*.npy"))
    regen += list((ROOT / "data" / "processed").glob("rowmask_*.npy"))
    if regen:
        tot = sum(f.stat().st_size for f in regen) / 1e6
        print(f"\nregenerable feature files: {tot:,.0f} MB across {len(regen)} files")
        if args.purge:
            for f in regen:
                f.unlink()
            print("deleted -- rerun featurize.py when you need them")
        else:
            print("run with --purge to delete them")


if __name__ == "__main__":
    main()
