"""Check that every number in the README traces to a committed result file.

The README's standing rule is that no performance number appears unless a script
in this repo produced it and the raw output is committed. This is the check.

**What it does not do, stated plainly.** It matches a decimal in the prose
against the text of everything under `results/`, so it proves a number *exists*
somewhere in the committed data — not that it came from the right experiment. An
earlier version of this check reported "all traceable" for five profile numbers
that had in fact been overwritten by a later run; they were matching by
coincidence inside an unrelated per-request CSV. That is why the report prints
*which* file each number was found in: a number backed only by a file that has
nothing to do with the claim is a finding, and only a human reading the pairing
can tell.

    python scripts/audit_numbers.py
    python scripts/audit_numbers.py --min-decimals 2 --show-all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md",)

# Numbers that are known not to trace, with the reason. Anything else unmatched
# is a failure. Keeping this list short and explicit is the point.
KNOWN_GAPS = {
    "8.5576": "milestone 2 late-arrival run, superseded in results/late_arrival.json by --mode labels",
    "9.5229": "same milestone 2 run",
    "10.300": "milestone 3 prose rounds static-headroom wall_s 10.2998",
}


def numbers(text: str, min_decimals: int) -> set[str]:
    return set(re.findall(rf"(?<![\w.])(\d+\.\d{{{min_decimals},}})", text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-decimals", type=int, default=3)
    ap.add_argument("--show-all", action="store_true", help="print every number and its source")
    args = ap.parse_args()

    sources = {
        p.relative_to(ROOT): p.read_text(errors="replace")
        for p in sorted((ROOT / "results").rglob("*"))
        if p.is_file()
    }
    if not sources:
        print("no files under results/")
        return 1

    failures = 0
    for doc in DOCS:
        text = (ROOT / doc).read_text()
        found: list[tuple[str, str]] = []
        unmatched: list[str] = []
        for n in sorted(numbers(text, args.min_decimals), key=float):
            hits = [str(f) for f, body in sources.items() if n in body]
            if hits:
                found.append((n, ", ".join(hits[:3]) + (" …" if len(hits) > 3 else "")))
            else:
                unmatched.append(n)

        print(f"\n{doc}: {len(found)} traced, {len(unmatched)} unmatched")
        if args.show_all:
            for n, where in found:
                print(f"    {n:>12}  <- {where}")
        for n in unmatched:
            reason = KNOWN_GAPS.get(n)
            if reason:
                print(f"    {n:>12}  KNOWN GAP: {reason}")
            else:
                print(f"    {n:>12}  UNTRACEABLE")
                failures += 1

    print("\nFAIL" if failures else "\nOK — every number traces, or is a documented gap")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
