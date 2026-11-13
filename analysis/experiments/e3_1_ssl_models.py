#!/usr/bin/env python3
"""
Question addressed: No tests on modern models? wav2vec 2.0, HuBERT and WavLM are cited
but not evaluated.

Constraint: Bridge2AI Voice v3.0.0 (PhysioNet) releases derived features only;
raw waveforms require a separate controlled-access request. Three routes,
run whichever data are available:

  Route A  Bridge2AI PPG (features/ppgs.parquet): phonetic posteriorgrams from a
           pretrained neural acoustic model -> learned representation already in
           the release. NOTE: the local copy is not a valid parquet file
           ("Parquet magic bytes not found"); re-download it from PhysioNet first.
  Route B  Bridge2AI raw audio (if access is granted): set --b2ai-audio-manifest
           to a CSV with participant_id, session_id, task (=raw_task),
           audio_path; embeddings from wav2vec2-base, hubert-base-ls960,
           wavlm-base-plus (layers 6 and 12, mean pooled) are extracted with
           common/audio_features.py and pushed through the IDENTICAL PCMVF
           pipeline (raw -> standardized speech AUC, same residual protocol).
  Route C  MDVR-KCL (local WAVs): same three SSL models; raw and PCMVF AUC per
           task (external_pcmvf.py). Small (n = 36), reported as supplementary.

Key question answered: does the protocol-sensitive / supported pattern depend on
the representation? The protocol model is unchanged, so only speech AUC can move.

Outputs (results/e3_1_ssl_models/): ssl_pcmvf_bridge2ai.csv (A and/or B),
mdvr/<model>/external_pcmvf_summary.csv (C)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402
from audio_features import to_lookup  # noqa: E402

HERE = Path(__file__).resolve().parent
SSL_MODELS = ["facebook/wav2vec2-base", "facebook/hubert-base-ls960", "microsoft/wavlm-base-plus"]


def register(data, name, lut_by_strippedpid):
    pid_map = {str(p).lstrip("0"): str(p) for p in data.manifest["participant_id"].unique()}
    data.lookups[name] = {(pid_map.get(k[0].lstrip("0"), k[0]), k[1], k[2]): v
                          for k, v in lut_by_strippedpid.items()}


def ppg_lookup(path: Path):
    """Released PPGs: column `ppgs` = list over 40 phoneme classes of per-frame posteriors
    (half precision). Summarised per recording as per-class mean + std over frames (80-D).
    Read one row group at a time to bound memory (file is ~1.9 GB)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    lut = {}
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["participant_id", "session_id", "task_name", "ppgs"])
        for pid, sid, task, ppg in zip(*(t.column(c).to_pylist() for c in ["participant_id", "session_id", "task_name"]),
                                       t.column("ppgs").to_pylist()):
            if not ppg:
                continue
            mat = np.asarray(ppg, dtype=np.float32)
            if mat.ndim != 2:
                continue
            if mat.shape[0] != 40 and mat.shape[1] == 40:
                mat = mat.T  # -> (40, frames)
            lut[(str(pid), str(sid), C.pb.norm_task(task))] = np.concatenate([mat.mean(1), mat.std(1)])
        print(f"[ppg] row group {g + 1}/{pf.num_row_groups}: {len(lut)} recordings", flush=True)
    return lut


def bridge2ai_rows(data, rep_names, args, diseases):
    rows = []
    for d in diseases:
        print(f"\n== {d}", flush=True)
        try:
            raw, ids, _ = C.raw_predictions(
                data, d, args, {n: (lambda n: lambda r, i, f: C.speech_matrix(r, i, data.lookups, [n], strict=False)[0])(n)
                                for n in rep_names} | {
                    "released_470d": lambda r, i, f: C.v6.build_speech_matrix_available(r, i, data.lookups)[0]},
                reps=rep_names + C.ORIGINAL_REPS)
            braw = C.pair_bootstrap(raw, None, args.bootstrap, args.seed + 303)
            # standardized: restrict recordings to those that have every representation
            man = data.manifest.copy()
            keys = list(zip(man["participant_id"], man["session_id"], man["task_norm"]))
            man["_has_all_reps"] = man["_has_all_reps"] & np.array(
                [all(k in data.lookups[n] for n in rep_names) for k in keys])
            co = C.standardized_cohort(data, d, args, manifest=man)
            fns = {n: C.speech_fn(data, [n]) for n in rep_names}
            fns["released_470d"] = C.speech_fn(data)
            fns["protocol"] = C.protocol_fn(data)
            std = C.standardized_predictions(data, co, args, fns)
            bstd = C.pair_bootstrap(std, co.final_pairs, args.bootstrap, args.seed + 5001)
            for n in rep_names + ["released_470d"]:
                rows.append({
                    "disease": d, "representation": n, "raw_n": braw["n"], "raw_auc": braw[f"{n}_auc"],
                    "raw_ci_low": braw[f"{n}_ci_low"], "raw_ci_high": braw[f"{n}_ci_high"],
                    "pairs": len(co.final_pairs), "std_speech_auc": bstd[f"{n}_auc"],
                    "std_ci_low": bstd[f"{n}_ci_low"], "std_ci_high": bstd[f"{n}_ci_high"],
                    "residual_protocol_auc": bstd["protocol_auc"],
                    "pcmvf_label": C.pcmvf_label(bstd[f"{n}_ci_low"], bstd["protocol_ci_low"],
                                                 bstd["protocol_ci_high"], co.tier)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"disease": d, "error": repr(exc)})
    return rows


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__), str(C.RESULTS / "e3_1_ssl_models"))
    p.add_argument("--routes", nargs="+", default=["A", "B", "C"])
    p.add_argument("--b2ai-audio-manifest", type=Path, default=None)
    p.add_argument("--models", nargs="+", default=SSL_MODELS)
    p.add_argument("--layers", nargs="+", default=["6", "12"])
    p.add_argument("--device", default="cuda")
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    py = sys.executable

    if {"A", "B"} & set(ns.routes):
        data = C.load_data(ns.b2ai_root, attach_metadata=False)
        reps = []
        if "A" in ns.routes:
            try:
                register(data, "ppg", ppg_lookup(data.b2ai_root / "features" / "ppgs.parquet"))
                reps.append("ppg")
            except Exception as exc:  # noqa: BLE001
                print(f"[route A skipped] ppgs.parquet unreadable: {exc!r}. Re-download from PhysioNet.")
        if "B" in ns.routes and ns.b2ai_audio_manifest:
            fdir = ns.out_dir / "b2ai_ssl_features"
            subprocess.run([py, str(C.COMMON_DIR / "audio_features.py"), "--manifest", str(ns.b2ai_audio_manifest),
                            "--features", *[f"ssl:{m}" for m in ns.models], "--layers", *ns.layers,
                            "--device", ns.device, "--out-dir", str(fdir)], check=True)
            for f in sorted(fdir.glob("ssl__*.parquet")):
                df = pd.read_parquet(f)
                df["task_norm"] = df["task"].map(C.pb.norm_task)  # match manifest task_norm convention
                register(data, f.stem, to_lookup(df))
                reps.append(f.stem)
        elif "B" in ns.routes:
            print("[route B skipped] no --b2ai-audio-manifest (Bridge2AI raw audio requires separate access).")
        if reps:
            C.save(bridge2ai_rows(data, reps, args, ns.diseases), ns.out_dir, "ssl_pcmvf_bridge2ai.csv")

    if "C" in ns.routes:
        cmd = [py, str(HERE / "e2_5_external_dataset_pcmvf.py"), "--dataset", "mdvr_kcl",
               "--features", "egemaps", *[f"ssl:{m}" for m in ns.models], "--layers", *ns.layers,
               "--device", ns.device]
        if ns.quick:
            cmd.append("--quick")
        print("[route C] results land in results/e2_5_external_dataset_pcmvf/mdvr_kcl/")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
