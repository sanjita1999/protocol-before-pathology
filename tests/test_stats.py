"""Smallest checks that fail if the PCMVF statistics or label rule break. No data needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "common"))
import pcmvf_common as C  # noqa: E402


def _preds(y, s):
    return pd.DataFrame({"participant_id": [str(i) for i in range(len(y))], "y_true": y, "y_score": s})


def test_labels():
    assert C.pcmvf_label(0.60, 0.55, 0.80, "primary_supported") == "protocol_sensitive"
    assert C.pcmvf_label(0.60, 0.40, 0.60, "primary_supported") == "observed_protocol_standardized_detectability_supported"
    assert C.pcmvf_label(0.45, 0.40, 0.60, "primary_supported") == "speech_not_resolved_after_standardization"
    assert C.pcmvf_label(0.60, 0.40, 0.60, "exploratory_small_n") == "exploratory_small_n"


def test_delong_and_bootstrap():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(50)].astype(int)
    strong, null = y + rng.normal(0, 0.7, 100), rng.normal(0, 1, 100)
    auc, se, lo, hi = C.delong_auc_ci(y, strong)
    assert lo > 0.5 and 0 < se < 0.1
    assert C.delong_paired_test(y, strong, null)["p"] < 0.01
    pairs = [(str(i), str(i + 50)) for i in range(50)]
    b = C.pair_bootstrap({"s": _preds(y, strong), "n": _preds(y, null)}, pairs, 300, 1, deltas=[("s", "n")])
    assert b["s_ci_low"] > 0.5 and b["n_ci_low"] < 0.5 < b["n_ci_high"] and b["delta[s-n]_ci_low"] > 0


def test_tost():
    draws = np.random.default_rng(1).normal(0.5, 0.02, 2000)
    assert C.tost_auc(draws, 0.5, 0.10)["equivalent"]
    assert not C.tost_auc(draws + 0.2, 0.7, 0.10)["equivalent"]
    assert 0.02 < C.smallest_equivalence_margin(draws) < 0.05


def test_pair_aware_folds_keep_pairs_together():
    pairs = [(f"c{i}", f"h{i}") for i in range(10)]
    ids = [x for p in pairs for x in p]
    f = C.v7.pair_aware_folds(pairs, ids, seed=0)
    assert all(f[2 * i] == f[2 * i + 1] for i in range(10))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
