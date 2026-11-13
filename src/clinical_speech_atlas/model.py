from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


@dataclass
class FoldPrediction:
    participant_id: str
    disease: str
    fold: int
    y_true: int
    y_score: float
    y_pred: int


class CovariateResidualizer:
    def __init__(self, context: str):
        self.context = context
        self.preprocessor = None
        self.regressor = None
        self.cols: list[str] = []

    def _columns(self) -> tuple[list[str], list[str]]:
        numeric, categorical = [], []
        if self.context in {"age", "age_sex", "joint"}:
            numeric.append("age")
        if self.context in {"sex", "age_sex", "joint"}:
            categorical.append("sex")
        if self.context in {"site", "joint"}:
            categorical.append("site")
        return numeric, categorical

    def fit(self, X: np.ndarray, cov: pd.DataFrame):
        numeric, categorical = self._columns()
        self.cols = numeric + categorical
        if not self.cols:
            return self
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("std", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical))
        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        Z = self.preprocessor.fit_transform(cov[self.cols])
        self.regressor = LinearRegression().fit(Z, X)
        return self

    def transform(self, X: np.ndarray, cov: pd.DataFrame) -> np.ndarray:
        if not self.cols:
            return X
        Z = self.preprocessor.transform(cov[self.cols])
        nuisance = self.regressor.predict(Z)
        return X - nuisance


def safe_pls_components(X: np.ndarray, y: np.ndarray, max_components: int) -> int:
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    class_counts = np.bincount(y.astype(int), minlength=2)
    minority = int(class_counts.min())
    return max(1, min(max_components, p, n - 1, max(rank, 1), max(minority - 1, 1)))


def _prepare_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cov_train: pd.DataFrame,
    cov_test: pd.DataFrame,
    context: str,
):
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)
    if context != "raw":
        residualizer = CovariateResidualizer(context).fit(X_train, cov_train)
        X_train = residualizer.transform(X_train, cov_train)
        X_test = residualizer.transform(X_test, cov_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def run_grouped_probe(
    df: pd.DataFrame,
    disease: str,
    cfg: dict[str, Any],
    context: str = "raw",
    probe: str = "logistic",
    fixed_pls_components: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    needed = {"participant_id", "cohort", "fold", "vector", "age", "sex", "site"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for probe: {sorted(missing)}")
    sub = df[df["cohort"].isin(["Healthy Control", disease])].copy()
    sub["y"] = sub["cohort"].eq(disease).astype(int)
    if sub["y"].nunique() < 2:
        raise ValueError(f"Need cases and controls for {disease}")
    preds = []
    for fold in sorted(sub["fold"].dropna().unique()):
        tr = sub["fold"].ne(fold)
        te = sub["fold"].eq(fold)
        train, test = sub.loc[tr].copy(), sub.loc[te].copy()
        ytr = train["y"].to_numpy(int)
        yte = test["y"].to_numpy(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            raise ValueError(f"Fold {fold} for {disease} lacks both classes")
        Xtr = np.stack(train["vector"].map(np.asarray).to_list()).astype(float)
        Xte = np.stack(test["vector"].map(np.asarray).to_list()).astype(float)
        Xtr, Xte = _prepare_fold(Xtr, Xte, train[["age", "sex", "site"]], test[["age", "sex", "site"]], context)
        k = fixed_pls_components or safe_pls_components(Xtr, ytr, cfg["analysis"]["pls_max_components"])
        k = min(k, safe_pls_components(Xtr, ytr, k))
        pls = PLSRegression(n_components=k, scale=False)
        pls.fit(Xtr, ytr)
        Ztr = pls.transform(Xtr)
        Zte = pls.transform(Xte)
        if probe == "logistic":
            clf = LogisticRegression(
                C=float(cfg["analysis"].get("logistic_C", 1.0)),
                class_weight="balanced",
                solver=cfg["analysis"].get("logistic_solver", "liblinear"),
                max_iter=int(cfg["analysis"].get("logistic_max_iter", 3000)),
                random_state=int(cfg["project"]["seed"]),
            )
            clf.fit(Ztr, ytr)
            score = clf.predict_proba(Zte)[:, 1]
        elif probe == "linear_svm":
            clf = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=int(cfg["project"]["seed"]))
            clf.fit(Ztr, ytr)
            score = clf.predict_proba(Zte)[:, 1]
        else:
            raise ValueError(probe)
        pred = (score >= 0.5).astype(int)
        for pid, yt, ys, yp in zip(test["participant_id"], yte, score, pred):
            preds.append(FoldPrediction(str(pid), disease, int(fold), int(yt), float(ys), int(yp)).__dict__)
    out = pd.DataFrame(preds)
    metrics = compute_metrics(out["y_true"].to_numpy(), out["y_score"].to_numpy())
    metrics.update({
        "disease": disease,
        "n_case": int(out["y_true"].sum()),
        "n_control": int((1 - out["y_true"]).sum()),
        "context": context,
        "probe": probe,
    })
    return out, metrics




def _balanced_resample_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """Deterministically oversample the minority class for estimators without class_weight.

    This is used only by the nonlinear MLP robustness probe.  Resampling is
    performed inside each training fold and never touches held-out participants.
    """
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Both classes are required for balanced resampling")
    n = max(len(idx0), len(idx1))
    a = rng.choice(idx0, size=n, replace=len(idx0) < n)
    b = rng.choice(idx1, size=n, replace=len(idx1) < n)
    out = np.concatenate([a, b])
    return out[rng.permutation(len(out))]


def run_mlp_probe(
    df: pd.DataFrame,
    disease: str,
    cfg: dict[str, Any],
    context: str = "raw",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Nonlinear participant-level robustness probe.

    Unlike the frozen primary PLS+logistic probe, the MLP operates directly on
    train-fold-imputed and standardized participant vectors.  Class imbalance
    is handled by deterministic minority-class oversampling inside each training
    fold.  This model is a robustness analysis only; it does not replace the
    preregistered/frozen primary probe.
    """
    needed = {"participant_id", "cohort", "fold", "vector", "age", "sex", "site"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for MLP probe: {sorted(missing)}")
    sub = df[df["cohort"].isin(["Healthy Control", disease])].copy()
    sub["y"] = sub["cohort"].eq(disease).astype(int)
    if sub["y"].nunique() < 2:
        raise ValueError(f"Need cases and controls for {disease}")

    rob = cfg.get("model_robustness", {})
    hidden = tuple(int(x) for x in rob.get("mlp_hidden_layers", [64, 32]))
    alpha = float(rob.get("mlp_alpha", 1e-3))
    max_iter = int(rob.get("mlp_max_iter", 500))
    lr = float(rob.get("mlp_learning_rate_init", 1e-3))
    seed0 = int(cfg["project"]["seed"])

    preds = []
    for fold in sorted(sub["fold"].dropna().unique()):
        train = sub[sub["fold"].ne(fold)].copy()
        test = sub[sub["fold"].eq(fold)].copy()
        ytr = train["y"].to_numpy(int)
        yte = test["y"].to_numpy(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            raise ValueError(f"Fold {fold} for {disease} lacks both classes")
        Xtr = np.stack(train["vector"].map(np.asarray).to_list()).astype(float)
        Xte = np.stack(test["vector"].map(np.asarray).to_list()).astype(float)
        Xtr, Xte = _prepare_fold(
            Xtr, Xte,
            train[["age", "sex", "site"]], test[["age", "sex", "site"]],
            context,
        )
        rs_seed = seed0 + 1009 * int(fold) + sum(ord(c) for c in disease)
        bi = _balanced_resample_indices(ytr, rs_seed)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=min(32, len(bi)),
            learning_rate_init=lr,
            max_iter=max_iter,
            shuffle=True,
            random_state=rs_seed,
            early_stopping=False,
        )
        clf.fit(Xtr[bi], ytr[bi])
        score = clf.predict_proba(Xte)[:, 1]
        for pid, yt, ys in zip(test["participant_id"], yte, score):
            preds.append({
                "participant_id": str(pid), "disease": disease, "fold": int(fold),
                "y_true": int(yt), "y_score": float(ys), "y_pred": int(ys >= 0.5),
            })
    out = pd.DataFrame(preds)
    m = compute_metrics(out.y_true.to_numpy(), out.y_score.to_numpy())
    m.update({
        "disease": disease, "n_case": int(out.y_true.sum()),
        "n_control": int((1 - out.y_true).sum()), "context": context,
        "probe": "mlp_raw_vector",
    })
    return out, m


def run_gated_attention_mil(
    df: pd.DataFrame,
    disease: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Gated-attention multiple-instance learning over recording embeddings.

    ``df`` must contain one row per participant with a ``bag`` column holding a
    2-D array [n_recordings, embedding_dim].  All imputation/scaling is fit on
    training-participant recordings only.  The model uses a fixed epoch budget
    selected a priori in config; held-out folds are never used for early
    stopping or hyperparameter selection.
    """
    try:
        import torch
        from torch import nn
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Gated-attention MIL requires PyTorch. Install torch in the active "
            "environment or run model-robustness with --skip-mil."
        ) from e

    needed = {"participant_id", "cohort", "fold", "bag"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for MIL probe: {sorted(missing)}")
    sub = df[df.cohort.isin(["Healthy Control", disease])].copy()
    sub["y"] = sub.cohort.eq(disease).astype(int)
    if sub.y.nunique() < 2:
        raise ValueError(f"Need cases and controls for {disease}")

    rob = cfg.get("model_robustness", {})
    hidden_dim = int(rob.get("mil_hidden_dim", 64))
    attn_dim = int(rob.get("mil_attention_dim", 32))
    epochs = int(rob.get("mil_epochs", 60))
    batch_size = int(rob.get("mil_batch_size", 16))
    lr = float(rob.get("mil_learning_rate", 1e-3))
    wd = float(rob.get("mil_weight_decay", 1e-4))
    dropout = float(rob.get("mil_dropout", 0.15))
    seed0 = int(cfg["project"]["seed"])
    device_name = str(rob.get("mil_device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    class GatedAttentionMIL(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.v = nn.Linear(hidden_dim, attn_dim)
            self.u = nn.Linear(hidden_dim, attn_dim)
            self.w = nn.Linear(attn_dim, 1, bias=False)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x, mask):
            # x: [B, N, D], mask: [B, N]
            h = self.proj(x)
            a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
            a = a.masked_fill(~mask, -1e9)
            a = torch.softmax(a, dim=1)
            z = torch.sum(h * a.unsqueeze(-1), dim=1)
            return self.head(z).squeeze(-1)

    def pad_batch(bags, ys):
        max_n = max(len(b) for b in bags)
        d = bags[0].shape[1]
        xb = np.zeros((len(bags), max_n, d), dtype=np.float32)
        mb = np.zeros((len(bags), max_n), dtype=bool)
        for i, b in enumerate(bags):
            xb[i, :len(b)] = b
            mb[i, :len(b)] = True
        return (
            torch.from_numpy(xb).to(device),
            torch.from_numpy(mb).to(device),
            torch.tensor(ys, dtype=torch.float32, device=device),
        )

    preds = []
    for fold in sorted(sub.fold.dropna().unique()):
        train = sub[sub.fold.ne(fold)].copy().reset_index(drop=True)
        test = sub[sub.fold.eq(fold)].copy().reset_index(drop=True)
        ytr = train.y.to_numpy(int)
        yte = test.y.to_numpy(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            raise ValueError(f"Fold {fold} for {disease} lacks both classes")

        # Train-only recording-level imputation and scaling.
        flat_train = np.concatenate([np.asarray(b, dtype=float) for b in train.bag], axis=0)
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        flat_train = imp.fit_transform(flat_train)
        scaler.fit(flat_train)

        def tx(b):
            arr = imp.transform(np.asarray(b, dtype=float))
            return scaler.transform(arr).astype(np.float32)

        train_bags = [tx(b) for b in train.bag]
        test_bags = [tx(b) for b in test.bag]
        input_dim = int(train_bags[0].shape[1])

        fold_seed = seed0 + 2027 * int(fold) + sum(ord(c) for c in disease)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        model = GatedAttentionMIL(input_dim).to(device)
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(int((1 - ytr).sum()), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        rng = np.random.default_rng(fold_seed)

        model.train()
        for _ in range(epochs):
            order = rng.permutation(len(train_bags))
            for start in range(0, len(order), batch_size):
                ids = order[start:start + batch_size]
                xb, mb, yb = pad_batch([train_bags[i] for i in ids], ytr[ids])
                opt.zero_grad(set_to_none=True)
                logits = model(xb, mb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(test_bags), batch_size):
                ids = np.arange(start, min(start + batch_size, len(test_bags)))
                xb, mb, _ = pad_batch([test_bags[i] for i in ids], yte[ids])
                score = torch.sigmoid(model(xb, mb)).detach().cpu().numpy()
                scores.extend(score.tolist())
        for pid, yt, ys in zip(test.participant_id, yte, scores):
            preds.append({
                "participant_id": str(pid), "disease": disease, "fold": int(fold),
                "y_true": int(yt), "y_score": float(ys), "y_pred": int(ys >= 0.5),
            })

    out = pd.DataFrame(preds)
    m = compute_metrics(out.y_true.to_numpy(), out.y_score.to_numpy())
    m.update({
        "disease": disease, "n_case": int(out.y_true.sum()),
        "n_control": int((1 - out.y_true).sum()), "context": "equal_budget_recording_bags",
        "probe": "gated_attention_mil",
    })
    return out, m

def run_covariate_only(df: pd.DataFrame, disease: str, context: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    sub = df[df["cohort"].isin(["Healthy Control", disease])].copy()
    sub["y"] = sub["cohort"].eq(disease).astype(int)
    if context == "age":
        numeric, categorical = ["age"], []
    elif context == "sex":
        numeric, categorical = [], ["sex"]
    elif context == "site":
        numeric, categorical = [], ["site"]
    elif context == "age_sex":
        numeric, categorical = ["age"], ["sex"]
    elif context == "joint":
        numeric, categorical = ["age"], ["sex", "site"]
    else:
        raise ValueError(context)
    preds = []
    for fold in sorted(sub["fold"].dropna().unique()):
        train = sub[sub["fold"].ne(fold)]
        test = sub[sub["fold"].eq(fold)]
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("std", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical))
        pre = ColumnTransformer(transformers)
        Xtr = pre.fit_transform(train[numeric + categorical])
        Xte = pre.transform(test[numeric + categorical])
        clf = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000, random_state=int(cfg["project"]["seed"]))
        clf.fit(Xtr, train["y"])
        score = clf.predict_proba(Xte)[:, 1]
        for pid, yt, ys in zip(test["participant_id"], test["y"], score):
            preds.append({"participant_id": str(pid), "disease": disease, "fold": int(fold), "y_true": int(yt), "y_score": float(ys), "y_pred": int(ys >= 0.5)})
    out = pd.DataFrame(preds)
    m = compute_metrics(out.y_true.to_numpy(), out.y_score.to_numpy())
    m.update({"disease": disease, "context": context, "n_case": int(out.y_true.sum()), "n_control": int((1-out.y_true).sum())})
    return out, m



def run_speech_plus_covariates(
    df: pd.DataFrame,
    disease: str,
    cfg: dict[str, Any],
    context: str = "age_sex",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit the frozen speech probe with demographics appended after PLS.

    Speech preprocessing (imputation, standardization, PLS) and demographic
    preprocessing are both fit strictly inside each training fold.  This model
    is used only for incremental-value analysis; it does not replace the frozen
    speech-only primary probe.
    """
    if context != "age_sex":
        raise ValueError("Incremental analysis currently supports context='age_sex' only")
    needed = {"participant_id", "cohort", "fold", "vector", "age", "sex", "site"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for speech+covariate probe: {sorted(missing)}")
    sub = df[df["cohort"].isin(["Healthy Control", disease])].copy()
    sub["y"] = sub["cohort"].eq(disease).astype(int)
    preds = []
    for fold in sorted(sub["fold"].dropna().unique()):
        train = sub[sub["fold"].ne(fold)].copy()
        test = sub[sub["fold"].eq(fold)].copy()
        ytr = train["y"].to_numpy(int)
        yte = test["y"].to_numpy(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            raise ValueError(f"Fold {fold} for {disease} lacks both classes")

        Xtr = np.stack(train["vector"].map(np.asarray).to_list()).astype(float)
        Xte = np.stack(test["vector"].map(np.asarray).to_list()).astype(float)
        Xtr, Xte = _prepare_fold(
            Xtr, Xte,
            train[["age", "sex", "site"]], test[["age", "sex", "site"]],
            "raw",
        )
        k = safe_pls_components(Xtr, ytr, int(cfg["analysis"]["pls_max_components"]))
        pls = PLSRegression(n_components=k, scale=False)
        pls.fit(Xtr, ytr)
        Ztr_speech = pls.transform(Xtr)
        Zte_speech = pls.transform(Xte)

        demo_pre = ColumnTransformer([
            ("age", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("std", StandardScaler()),
            ]), ["age"]),
            ("sex", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ["sex"]),
        ])
        Ztr_demo = demo_pre.fit_transform(train[["age", "sex"]])
        Zte_demo = demo_pre.transform(test[["age", "sex"]])
        Ztr = np.column_stack([Ztr_speech, np.asarray(Ztr_demo, dtype=float)])
        Zte = np.column_stack([Zte_speech, np.asarray(Zte_demo, dtype=float)])

        clf = LogisticRegression(
            C=float(cfg["analysis"].get("logistic_C", 1.0)),
            class_weight="balanced",
            solver=cfg["analysis"].get("logistic_solver", "liblinear"),
            max_iter=int(cfg["analysis"].get("logistic_max_iter", 3000)),
            random_state=int(cfg["project"]["seed"]),
        )
        clf.fit(Ztr, ytr)
        score = clf.predict_proba(Zte)[:, 1]
        for pid, yt, ys in zip(test["participant_id"], yte, score):
            preds.append({
                "participant_id": str(pid), "disease": disease, "fold": int(fold),
                "y_true": int(yt), "y_score": float(ys), "y_pred": int(ys >= 0.5),
            })

    out = pd.DataFrame(preds)
    metrics = compute_metrics(out.y_true.to_numpy(), out.y_score.to_numpy())
    metrics.update({
        "disease": disease, "context": "speech_age_sex",
        "n_case": int(out.y_true.sum()), "n_control": int((1 - out.y_true).sum()),
    })
    return out, metrics


def run_tabular_probe(
    df: pd.DataFrame,
    disease: str,
    cfg: dict[str, Any],
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    context: str = "tabular",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Leakage-controlled logistic baseline for non-speech participant features."""
    categorical_cols = list(categorical_cols or [])
    needed = {"participant_id", "cohort", "fold", *numeric_cols, *categorical_cols}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for tabular probe: {sorted(missing)}")
    sub = df[df.cohort.isin(["Healthy Control", disease])].copy()
    sub["y"] = sub.cohort.eq(disease).astype(int)
    preds = []
    for fold in sorted(sub.fold.dropna().unique()):
        train = sub[sub.fold.ne(fold)].copy()
        test = sub[sub.fold.eq(fold)].copy()
        if train.y.nunique() < 2 or test.y.nunique() < 2:
            raise ValueError(f"Fold {fold} for {disease} lacks both classes")
        transformers = []
        if numeric_cols:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("std", StandardScaler()),
            ]), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols))
        pre = ColumnTransformer(transformers, remainder="drop")
        Xtr = pre.fit_transform(train[numeric_cols + categorical_cols])
        Xte = pre.transform(test[numeric_cols + categorical_cols])
        clf = LogisticRegression(
            C=float(cfg["analysis"].get("logistic_C", 1.0)),
            class_weight="balanced",
            solver=cfg["analysis"].get("logistic_solver", "liblinear"),
            max_iter=int(cfg["analysis"].get("logistic_max_iter", 3000)),
            random_state=int(cfg["project"]["seed"]),
        )
        clf.fit(Xtr, train.y.to_numpy(int))
        score = clf.predict_proba(Xte)[:, 1]
        for pid, yt, ys in zip(test.participant_id, test.y, score):
            preds.append({
                "participant_id": str(pid), "disease": disease, "fold": int(fold),
                "y_true": int(yt), "y_score": float(ys), "y_pred": int(ys >= 0.5),
            })
    out = pd.DataFrame(preds)
    metrics = compute_metrics(out.y_true.to_numpy(), out.y_score.to_numpy())
    metrics.update({
        "disease": disease, "context": context,
        "n_case": int(out.y_true.sum()), "n_control": int((1 - out.y_true).sum()),
    })
    return out, metrics

def compute_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }
