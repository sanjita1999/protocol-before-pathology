#!/usr/bin/env python3
"""
Question addressed: Position novelty relative to Shukla et al. (2026), which already
shows demographic (age, country) confounding on the same Bridge2AI release. The
distinctive contribution is modelling assessment STRUCTURE itself.

Quantifies how much diagnostic information the assessment structure carries
BEYOND the demographic confounders Shukla et al. studied:

  demo_age_sex           age + sex
  demo_age_sex_country   age + sex + country (Shukla-style confounders)
  structure_only         completion + per-family counts + session/observation totals
                         (no demographics, no country)
  structure_no_cog       structure_only after deleting cognitive-linguistic rows
  full_protocol          paper's protocol model (structure + age + sex)
  full_plus_country      full_protocol + country

Paired (participant-bootstrap) deltas:
  structure_only - demo_age_sex_country
  full_plus_country - demo_age_sex_country   (incremental value of structure)
Repeated in the USA-only subset, where country is constant by construction.

Outputs (results/e2_6_structure_beyond_demographics/): structure_vs_demographics.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e2_6_structure_beyond_demographics"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    meta = data.participant_meta
    out = []

    def feats(r, i, f):
        sub = r[r["participant_id"].isin(i)]
        P = C.protocol_df(data, sub, i, f)
        P_nc = C.protocol_df(data, sub[sub["task_family"] != "cognitive_linguistic"], i,
                             [x for x in f if x != "cognitive_linguistic"])
        ctry = meta.reindex(i)["country_canada"].rename("country_canada").to_frame().set_index(P.index)
        struct = [c for c in P if c not in ("age", "sex_m")]
        return {
            "demo_age_sex": P[["age", "sex_m"]],
            "demo_age_sex_country": pd.concat([P[["age", "sex_m"]], ctry], axis=1),
            "structure_only": P[struct],
            "structure_no_cog": P_nc[[c for c in P_nc if c not in ("age", "sex_m")]],
            "full_protocol": P,
            "full_plus_country": pd.concat([P, ctry], axis=1),
        }

    names = ["demo_age_sex", "demo_age_sex_country", "structure_only", "structure_no_cog",
             "full_protocol", "full_plus_country"]
    deltas = [("structure_only", "demo_age_sex_country"), ("full_plus_country", "demo_age_sex_country"),
              ("structure_no_cog", "demo_age_sex")]

    def run(d, dat, subset):
        fns = {n: (lambda n: lambda r, i, f: feats(r, i, f)[n].to_numpy(float))(n) for n in names}
        preds, ids, _ = C.raw_predictions(dat, d, args, fns)
        b = C.pair_bootstrap(preds, None, args.bootstrap, args.seed + 404, deltas=deltas)
        row = {"disease": d, "subset": subset, "n": b["n"]}
        row.update({k: v for k, v in b.items() if k not in ("n", "n_pairs")})
        return row

    def one(d):
        rows = [run(d, data, "all")]
        usa = set(meta.index[meta["country"].fillna("").eq("USA")])
        dusa = C.Data(data.manifest[data.manifest["participant_id"].isin(usa)].copy(),
                      data.cohort_by_pid[data.cohort_by_pid.index.isin(usa)],
                      data.demographics[data.demographics["participant_id"].astype(str).isin(usa)],
                      data.lookups, None, data.b2ai_root, meta.loc[meta.index.isin(usa)])
        try:
            rows.append(run(d, dusa, "usa_only"))
        except Exception as exc:  # noqa: BLE001
            rows.append({"disease": d, "subset": "usa_only", "error": repr(exc)})
        return rows

    out = C.run_each(ns.diseases, one)
    C.save(out, ns.out_dir, "structure_vs_demographics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
