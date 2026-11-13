"""Matched-pair bootstrap helper for PCMVF analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def paired_match_auc_ci(
    speech_pred: pd.DataFrame,
    protocol_pred: pd.DataFrame,
    matched_pairs,
    n_boot: int = 2000,
    seed: int = 42,
):
    """Bootstrap matched case-control pairs using fixed OOF predictions."""
    s = speech_pred[["participant_id", "y_true", "y_score"]].copy()
    p = protocol_pred[["participant_id", "y_true", "y_score"]].copy()

    s = s.rename(columns={"y_score": "speech_score"})
    p = p.rename(columns={"y_score": "protocol_score"})

    s["participant_id"] = s["participant_id"].astype(str)
    p["participant_id"] = p["participant_id"].astype(str)

    df = s.merge(
        p[["participant_id", "y_true", "protocol_score"]].rename(
            columns={"y_true": "protocol_y_true"}
        ),
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(s) or len(df) != len(p):
        raise RuntimeError(
            "Speech and protocol prediction tables do not contain the same participants."
        )

    if not (
        df["y_true"].astype(int).to_numpy()
        == df["protocol_y_true"].astype(int).to_numpy()
    ).all():
        raise RuntimeError("Speech and protocol labels disagree.")

    pair_rows = []
    for pair_id, pair in enumerate(matched_pairs):
        if len(pair) != 2:
            raise RuntimeError(
                f"Matched pair {pair_id} does not contain exactly 2 participants."
            )
        for participant_id in pair:
            pair_rows.append(
                {"participant_id": str(participant_id), "pair_id": pair_id}
            )

    pair_df = pd.DataFrame(pair_rows)

    df = df.merge(
        pair_df,
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )

    n_pairs = len(matched_pairs)

    if len(df) != 2 * n_pairs:
        raise RuntimeError(
            f"Expected {2 * n_pairs} matched participants but found {len(df)}."
        )

    for pair_id, g in df.groupby("pair_id", sort=False):
        labels = sorted(g["y_true"].astype(int).tolist())
        if len(g) != 2 or labels != [0, 1]:
            raise RuntimeError(
                f"Pair {pair_id} is invalid: n={len(g)}, labels={labels}"
            )

    speech_auc = float(roc_auc_score(df["y_true"], df["speech_score"]))
    protocol_auc = float(roc_auc_score(df["y_true"], df["protocol_score"]))

    pair_groups = {
        pair_id: g for pair_id, g in df.groupby("pair_id", sort=False)
    }
    pair_ids = np.asarray(list(pair_groups.keys()))
    rng = np.random.default_rng(seed)

    speech_boot = np.empty(n_boot, dtype=float)
    protocol_boot = np.empty(n_boot, dtype=float)
    delta_boot = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sampled_pair_ids = rng.choice(pair_ids, size=n_pairs, replace=True)

        boot_df = pd.concat(
            [pair_groups[pair_id] for pair_id in sampled_pair_ids],
            ignore_index=True,
        )

        yb = boot_df["y_true"].astype(int)
        s_auc = float(roc_auc_score(yb, boot_df["speech_score"]))
        p_auc = float(roc_auc_score(yb, boot_df["protocol_score"]))

        speech_boot[b] = s_auc
        protocol_boot[b] = p_auc
        delta_boot[b] = s_auc - p_auc

    speech_lo, speech_hi = np.percentile(speech_boot, [2.5, 97.5])
    protocol_lo, protocol_hi = np.percentile(protocol_boot, [2.5, 97.5])
    delta_lo, delta_hi = np.percentile(delta_boot, [2.5, 97.5])

    return {
        "n_pairs": int(n_pairs),
        "speech_auc": speech_auc,
        "speech_ci_low": float(speech_lo),
        "speech_ci_high": float(speech_hi),
        "protocol_auc": protocol_auc,
        "protocol_ci_low": float(protocol_lo),
        "protocol_ci_high": float(protocol_hi),
        "delta_auc": float(speech_auc - protocol_auc),
        "delta_ci_low": float(delta_lo),
        "delta_ci_high": float(delta_hi),
    }
