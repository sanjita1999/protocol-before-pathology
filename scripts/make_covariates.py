#!/usr/bin/env python3
"""
Build data/covariates.csv (participant_id, age_numeric, sex_clean) from the
public Bridge2AI Voice v3.0.0 demographics table.

Age "90 and above" (the release's top-coding) is left missing, exactly as in the
published analysis; those participants therefore cannot be age-matched. Sex is taken from
`sex_at_birth`. Participant ids have leading zeros stripped to match the
feature tables.

Usage:  python scripts/make_covariates.py --b2ai-root $B2AI_ROOT
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b2ai-root", default=os.environ.get("B2AI_ROOT"), required=os.environ.get("B2AI_ROOT") is None)
    p.add_argument("--out", default=str(ROOT / "data" / "covariates.csv"))
    ns = p.parse_args()
    d = pd.read_csv(Path(ns.b2ai_root) / "phenotype" / "demographics" / "demographics.tsv",
                    sep="\t", dtype={"participant_id": str}, low_memory=False)
    d["participant_id"] = d["participant_id"].str.lstrip("0")
    d["age_numeric"] = pd.to_numeric(d["age"], errors="coerce")  # "90 and above" -> NaN
    d["sex_clean"] = d["sex_at_birth"]
    out = d.groupby("participant_id", as_index=False)[["age_numeric", "sex_clean"]].first()
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ns.out, index=False)
    print(f"[saved] {ns.out} ({len(out)} participants, {out['age_numeric'].notna().sum()} with age)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
