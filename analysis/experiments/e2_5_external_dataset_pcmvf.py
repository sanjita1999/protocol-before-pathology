#!/usr/bin/env python3
"""
Question addressed: Demonstrate that PCMVF detects known collection artifacts in at
least one fully independent dataset.  Also applies to other modalities.

Builds a recording-level manifest for an independent dataset, extracts features,
and runs the generic PCMVF runner (common/external_pcmvf.py).

Datasets
  mdvr_kcl  (local, ready)  36 participants, 2 tasks, 1 session each, no age/sex.
            Protocol is balanced by design -> expected NEGATIVE control: Detect
            should give protocol AUC ~0.5 and PCMVF should not change speech AUC.
  mpower    (Synapse syn4993293, controlled access; recommended POSITIVE case)
            PD vs non-PD smartphone voice ("aaaah"), tapping, walking; participants
            contribute very different numbers of records and PD participants were
            prompted around medication timing; age differs strongly. Known record-
            vs subject-wise / confounding artifacts (Chaibub Neto et al., 2019,
            npj Digit Med; Neto et al. 2017). Detect should flag record counts and
            age; the same runner applies to the tapping/walking modalities.
            Provide --mpower-table: CSV with healthCode, recordId, createdOn,
            professional-diagnosis, age, gender, audio_path (downloaded .m4a->.wav)
  coswara   (public GitHub iiscleap/Coswara-Data) cough/breath/counting/vowel
            tasks, COVID status; task completion and recording counts vary.
            Provide --coswara-root (extracted folders with metadata CSV).

Example
  python e2_5_external_dataset_pcmvf.py --dataset mdvr_kcl --features egemaps ssl:microsoft/wavlm-base-plus
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
MDVR_ROOT = Path(os.environ.get("MDVR_ROOT", "data/external/MDVR-KCL"))


def mdvr_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for task_dir in ["ReadText", "SpontaneousDialogue"]:
        for grp in ["HC", "PD"]:
            for f in sorted((root / task_dir / grp).glob("*.wav")):
                parts = f.stem.split("_")  # ID02_pd_2_0_0 -> id, group, H&Y, UPDRS II-5, UPDRS III-18
                rows.append({"participant_id": parts[0], "session_id": "s1", "task": task_dir,
                             "label": int(parts[1] == "pd"), "audio_path": str(f)})
    return pd.DataFrame(rows)


def mpower_manifest(table: Path) -> pd.DataFrame:
    t = pd.read_csv(table)
    t["createdOn"] = pd.to_datetime(t["createdOn"], errors="coerce")
    return pd.DataFrame({
        "participant_id": t["healthCode"].astype(str),
        "session_id": t["createdOn"].dt.date.astype(str),   # one session = one day
        "task": "sustained_phonation",
        "label": t["professional-diagnosis"].astype(str).str.lower().isin(["true", "1"]).astype(int),
        "age": t["age"], "sex": t["gender"].astype(str).str.lower().str[0],
        "audio_path": t["audio_path"],
    })


IPVS_FAMILY = {"B": "text_reading", "PR": "text_reading", "FB": "phrase_word_reading",
               "D": "syllable_ddk", "V": "sustained_vowel"}


def _ipvs_code_demo(stem: str):
    """File code ...<birth yy><M|F><ddmm><yy|yyyy>...: returns (sex, age) or (None, None)."""
    import re
    m = re.search(r"(\d{2})([MF])(\d{8,})", stem)
    if not m:
        return None, None
    born, sex, d = int(m.group(1)), m.group(2).lower(), m.group(3)
    year = int(d[4:8]) if d[4:6] == "20" and len(d) >= 10 else 2000 + int(d[4:6])
    return sex, year - (1900 + born)


def ipvs_manifest(root: Path, controls: str) -> pd.DataFrame:
    """root = dataset/ipvs/raw. controls = 'elderly' (primary) or 'young' (known-artifact check).
    Age/sex: file code for PD and elderly controls; Excel sheet for young controls
    (their file codes reuse one template). Participant id includes the sub-folder
    because names repeat across PD sub-folders."""
    import re
    groups = {"pd": "28 People with Parkinson's disease",
              "elderly": "22 Elderly Healthy Control", "young": "15 Young Healthy Control"}
    x = pd.read_excel(root / groups["young"] / "15 YHC.xlsx", header=1)
    x.columns = [str(c).strip().lower() for c in x.columns]
    young = {f"{str(a).strip().lower()} {str(b).strip()[:1].lower()}": (str(sx).strip()[:1].lower(), ag)
             for a, b, sx, ag in zip(x["name"], x["surname"], x["sex"], x["age"])}
    rows = []
    for g in ("pd", controls):
        base = root / groups[g]
        for wav in base.rglob("*.wav"):
            m = re.match(r"^([A-Z]+?)(\d+)", wav.stem)
            if not m:
                continue
            if g == "young":
                parts = wav.parent.name.lower().split()
                sex, age = young.get(f"{parts[0]} {parts[-1][:1]}", (None, None))
            else:
                sex, age = _ipvs_code_demo(wav.stem)
            rows.append({"participant_id": f"{g}:{wav.parent.relative_to(base)}", "session_id": "s1",
                         "task": m.group(1) + m.group(2),
                         "task_family": IPVS_FAMILY.get(m.group(1)[:2], IPVS_FAMILY.get(m.group(1)[:1], m.group(1))),
                         "label": int(g == "pd"), "age": age, "sex": sex, "audio_path": str(wav)})
    df = pd.DataFrame(rows)
    # one age/sex per participant (first recording's code)
    first = df.groupby("participant_id")[["age", "sex"]].first()
    return df.drop(columns=["age", "sex"]).join(first, on="participant_id")


def coswara_manifest(root: Path) -> pd.DataFrame:
    """root = Coswara dir with repo/combined_data.csv and extracted/<date>/<id>/*.wav.
    Cases = positive_* ; controls = 'healthy' only (exposed / recovered / unverified excluded)."""
    meta = pd.read_csv(root / "repo" / "combined_data.csv").set_index("id")
    rows = []
    for wav in (root / "extracted").glob("*/*/*.wav"):  # single pass
        pid = wav.parent.name
        if pid not in meta.index:
            continue
        st = str(meta.at[pid, "covid_status"])
        if not (st.startswith("positive") or st == "healthy"):
            continue
        rows.append({"participant_id": pid, "session_id": "s1", "task": wav.stem,
                     "task_family": wav.stem.split("-")[0],  # breathing / cough / counting / vowel
                     "label": int(st.startswith("positive")), "age": meta.at[pid, "a"],
                     "sex": str(meta.at[pid, "g"])[:1].lower(), "audio_path": str(wav)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mdvr_kcl", "mpower", "coswara", "ipvs_elderly", "ipvs_young"], default="mdvr_kcl")
    p.add_argument("--features", nargs="+", default=["egemaps"])
    p.add_argument("--mpower-table", type=Path)
    p.add_argument("--coswara-root", type=Path, default=Path(os.environ.get("COSWARA_ROOT", "data/external/coswara")))
    p.add_argument("--mdvr-root", type=Path, default=MDVR_ROOT)
    p.add_argument("--ipvs-root", type=Path, default=Path(os.environ.get("IPVS_ROOT", "data/external/ipvs/raw")))
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", nargs="+", default=["6", "12"])
    p.add_argument("--min-pairs", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    ns = p.parse_args()
    out = C.RESULTS / "e2_5_external_dataset_pcmvf" / ns.dataset
    out.mkdir(parents=True, exist_ok=True)

    man = {"mdvr_kcl": lambda: mdvr_manifest(ns.mdvr_root),
           "mpower": lambda: mpower_manifest(ns.mpower_table),
           "coswara": lambda: coswara_manifest(ns.coswara_root),
           "ipvs_elderly": lambda: ipvs_manifest(ns.ipvs_root, "elderly"),
           "ipvs_young": lambda: ipvs_manifest(ns.ipvs_root, "young")}[ns.dataset]()
    man_path = out / "manifest.csv"
    man.to_csv(man_path, index=False)
    print(f"[manifest] {len(man)} recordings, {man['participant_id'].nunique()} participants -> {man_path}")

    py = sys.executable
    feat_dir = out / "features"
    subprocess.run([py, str(COMMON / "audio_features.py"), "--manifest", str(man_path), "--features",
                    *ns.features, "--layers", *ns.layers, "--device", ns.device, "--out-dir", str(feat_dir)],
                   check=True)
    extra = ["--seeds", "2", "--bootstrap", "200"] if ns.quick else []
    for f in sorted(feat_dir.glob("*.parquet")):
        subprocess.run([py, str(COMMON / "external_pcmvf.py"), "--manifest", str(man_path), "--features", str(f),
                        "--min-pairs", str(ns.min_pairs), "--out-dir", str(out / f.stem), *extra], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
