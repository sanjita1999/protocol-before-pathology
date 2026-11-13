#!/usr/bin/env python3
"""
Recording-level feature extraction from WAV files for datasets where raw audio
is available (MDVR-KCL locally; mPower / Coswara / Bridge2AI raw audio if access
is granted). Used by e2_5 and e3_1.

Input manifest CSV columns (required): participant_id, session_id, task, audio_path
Output parquet columns: participant_id, session_id, task, task_norm, vector (list[float]),
                        plus for SSL: model, layer

Feature sets
  egemaps          openSMILE eGeMAPSv02 functionals (88-D)          pip install opensmile
  ssl:<hf-model>   mean-pooled hidden states from a HuggingFace speech model,
                   one parquet per layer (or --layers), e.g.
                     ssl:facebook/wav2vec2-base
                     ssl:facebook/hubert-base-ls960
                     ssl:microsoft/wavlm-base-plus
                   pip install torch torchaudio transformers soundfile

Example
  python audio_features.py --manifest mdvr_manifest.csv --features ssl:microsoft/wavlm-base-plus \
      --layers 6 12 --out-dir results/features/mdvr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def norm_task(x) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def load_audio(path: str, sr: int = 16000, max_seconds: float | None = 60.0) -> np.ndarray:
    import soundfile as sf
    wav, fs = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if fs != sr:
        import torch
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), fs, sr).numpy()
    if max_seconds:
        wav = wav[: int(max_seconds * sr)]
    return wav


def extract_egemaps(man: pd.DataFrame) -> pd.DataFrame:
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    out = []
    for r in man.itertuples(index=False):
        try:
            v = smile.process_file(r.audio_path).to_numpy().ravel().astype(float)
        except Exception as exc:  # noqa: BLE001  (crowd-sourced audio: empty/corrupt files)
            print(f"[egemaps] skip {r.audio_path}: {exc}", flush=True)
            continue
        out.append({"participant_id": str(r.participant_id), "session_id": str(r.session_id),
                    "task": r.task, "task_norm": norm_task(r.task), "vector": v.tolist()})
    return pd.DataFrame(out)


def extract_ssl(man: pd.DataFrame, model_name: str, layers: list[int] | None, device: str,
                chunk_seconds: float = 20.0) -> dict[int, pd.DataFrame]:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
    n_layers = model.config.num_hidden_layers
    layers = layers or list(range(n_layers + 1))
    rows: dict[int, list] = {l: [] for l in layers}
    for i, r in enumerate(man.itertuples(index=False)):
        wav = load_audio(r.audio_path)
        step = int(chunk_seconds * 16000)
        sums, n = None, 0
        for s in range(0, max(len(wav), 1), step):
            seg = wav[s:s + step]
            if len(seg) < 400:
                continue
            inp = fe(seg, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                hs = model(**inp).hidden_states
            pooled = [h[0].mean(0).cpu().numpy() * h.shape[1] for h in hs]
            sums = pooled if sums is None else [a + b for a, b in zip(sums, pooled)]
            n += hs[0].shape[1]
        if sums is None:
            continue
        for l in layers:
            rows[l].append({"participant_id": str(r.participant_id), "session_id": str(r.session_id),
                            "task": r.task, "task_norm": norm_task(r.task), "model": model_name,
                            "layer": l, "vector": (sums[l] / n).astype(float).tolist()})
        if i % 50 == 0:
            print(f"[ssl] {model_name} {i + 1}/{len(man)}", flush=True)
    return {l: pd.DataFrame(v) for l, v in rows.items()}


def to_lookup(df: pd.DataFrame) -> dict:
    return {(str(a), str(b), str(c)): np.asarray(v, float)
            for a, b, c, v in zip(df["participant_id"], df["session_id"], df["task_norm"], df["vector"])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    ns = p.parse_args()
    man = pd.read_csv(ns.manifest, dtype=str)
    out = Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for feat in ns.features:
        if feat == "egemaps":
            extract_egemaps(man).to_parquet(out / "egemaps.parquet")
            print(f"[saved] {out / 'egemaps.parquet'}")
        elif feat.startswith("ssl:"):
            name = feat[4:]
            for layer, df in extract_ssl(man, name, ns.layers, ns.device).items():
                f = out / f"ssl__{name.replace('/', '__')}__L{layer}.parquet"
                df.to_parquet(f)
                print(f"[saved] {f}")
        else:
            raise ValueError(feat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
