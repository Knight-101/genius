"""Outcome-trained, monotone credit model; synthetic demonstration, not a validated credit rating.

Financial measures enter once. Recovery features contain only completed episodes, never
Grit's overlapping capacity score. Data quality is a publication gate, not a reward.
12-month outcomes must have matured before each training/selection boundary.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from . import config as C

FINANCIAL = [m for m in C.MEASURES if C.M[m]["area"] != "Data integrity"]
RECOVERY = ["gap_regained", "stayed_recovered", "funded_by_operations", "response", "weathered"]
VERSION = "Genius 5.1 · separate financial and resilience scores"
PREVIOUS_VERSION = "Genius 5 · learned financial and recovery model"


def features(ratings, grit):
    shape = next(iter(ratings.values())).shape
    cols = [1 - np.asarray(ratings[m], float) / 100 for m in FINANCIAL]
    nr = np.asarray(grit.get("recovery_n", np.zeros(shape)), float)
    evidence = nr / (nr + 2)
    parts = grit.get("recovery_features", {})
    for k in RECOVERY[:-1]:
        v = np.asarray(parts.get(k, np.full(shape, np.nan)), float)
        cols.append(np.where(np.isfinite(v), evidence * (0.5 - v), 0.0))
    nw = np.asarray(grit.get("weathered_n", np.zeros(shape)), float)
    cols.append(-nw / (nw + 2))
    return np.stack(cols, -1)


def _fit(X, y, penalty):
    """All inputs point toward higher risk; nonnegative slopes preserve direction."""
    p = float(np.clip(np.mean(y), 1e-5, 1 - 1e-5))
    initial = np.r_[np.log(p / (1 - p)), np.zeros(X.shape[1])]

    def objective(theta):
        z = theta[0] + X @ theta[1:]
        loss = np.mean(np.logaddexp(0, z) - y * z) + penalty * np.sum(theta[1:] ** 2) / 2
        error = expit(z) - y
        grad = np.r_[error.mean(), X.T @ error / len(y) + penalty * theta[1:]]
        return loss, grad

    with threadpool_limits(limits=2):
        res = minimize(objective, initial, jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] + [(0, 30)] * X.shape[1],
                       options={"maxiter": 350, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Credit model did not converge: {res.message}")
    return res.x


def metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                observations=int(len(y)), defaults=int(y.sum()))


@dataclass
class CreditModel:
    fills: np.ndarray
    theta: np.ndarray
    with_recovery: bool
    low: float
    high: float
    report: dict = field(default_factory=dict)
    version: str = VERSION

    def predict(self, ratings, grit):
        X = features(ratings, grit)
        X = np.where(np.isfinite(X), X, self.fills)
        use = X if self.with_recovery else X[..., :len(FINANCIAL)]
        z = self.theta[0] + use @ self.theta[1:]
        # A fixed monotone display scale anchored to training data; not 100 minus PD.
        score = np.clip(100 * (self.high - z) / max(self.high - self.low, 1e-6), 0, 100)
        recovery = (use[..., len(FINANCIAL):] @ self.theta[1 + len(FINANCIAL):]
                    if self.with_recovery else np.zeros(z.shape))
        return score, expit(z), recovery

    def apply(self, gs, ratings, grit):
        score, pd, recovery = self.predict(ratings, grit)
        quality = gs["areas"]["Data integrity"]
        eligible = gs["published"] & np.isfinite(quality) & (quality >= 60) & ~gs["ko"]["K5"]
        for k, cap in {"K1": 44, "K2": 59, "K3": 59, "K4": 59}.items():
            score = np.where(gs["ko"][k], np.minimum(score, cap), score)
        gs["rule_score"] = gs["score"].copy()
        gs["score"] = np.where(eligible, score, np.nan).astype(np.float32)
        gs["published"] = eligible
        gs["model_pd"] = np.where(eligible, pd, np.nan).astype(np.float32)
        gs["recovery_logodds"] = recovery.astype(np.float32)
        gs["model_version"] = self.version
        return gs

    def weights(self):
        names = FINANCIAL + (RECOVERY if self.with_recovery else [])
        total = max(float(self.theta[1:].sum()), 1e-12)
        rows = []
        for k, b in zip(names, self.theta[1:]):
            rows.append(dict(factor=C.M[k]["name"] if k in C.M else "Recovery: " + k.replace("_", " "),
                             area=C.M[k]["area"] if k in C.M else "Demonstrated recovery",
                             coefficient=float(b), relative_weight_pct=float(100 * b / total)))
        return sorted(rows, key=lambda r: -r["coefficient"])


def train(ratings, grit, default12, eligible, log=print):
    X = features(ratings, grit)
    T = X.shape[1]
    months = np.arange(T)
    # Fit boundary 59: default labels for observations through 47 have 12 months to mature.
    # Selection boundary 95: use observations 60..83. Final test 96..107 is never selected on.
    masks = dict(train=(months >= 11) & (months <= 47),
                 validation=(months >= 60) & (months <= 83), test=(months >= 96) & (months <= T - 13))
    rng = np.random.default_rng(20261029)
    data = {}
    fills = None
    for split, mask in masks.items():
        ok = eligible[:, mask] & np.isfinite(default12[:, mask])
        xx, yy = X[:, mask][ok], default12[:, mask][ok].astype(float)
        if len(yy) > 80000:
            ix = np.sort(rng.choice(len(yy), 80000, replace=False)); xx, yy = xx[ix], yy[ix]
        if split == "train":
            with np.errstate(all="ignore"):
                fills = np.nanmedian(xx, axis=0)
            fills = np.nan_to_num(fills, nan=0.5)
            fills[len(FINANCIAL):] = 0.0
        data[split] = (np.where(np.isfinite(xx), xx, fills), yy)
        if len(np.unique(yy)) < 2:
            raise ValueError(f"Both outcome classes are required in {split}")
    candidates = []
    for recovery in (False, True):
        for penalty in (0.0001, 0.001, 0.01):
            xx, yy = data["train"]
            theta = _fit(xx if recovery else xx[:, :len(FINANCIAL)], yy, penalty)
            vx, vy = data["validation"]
            pred = expit(theta[0] + (vx if recovery else vx[:, :len(FINANCIAL)]) @ theta[1:])
            candidates.append(dict(recovery=recovery, penalty=penalty, theta=theta, validation=metrics(vy, pred)))
    base = min((c for c in candidates if not c["recovery"]), key=lambda c: c["validation"]["log_loss"])
    combined = min((c for c in candidates if c["recovery"]), key=lambda c: c["validation"]["log_loss"])
    # Recovery must improve both ranking and probability error on the selection period.
    accepted = (combined["validation"]["brier"] < base["validation"]["brier"] * 0.999
                and combined["validation"]["auc"] > base["validation"]["auc"] + 0.001)
    candidate_passed = bool(accepted)
    accepted = False  # Separate scores by product design; never blend Grit into Genius.
    selected = base
    xx, _ = data["train"]
    theta = selected["theta"]
    z = theta[0] + (xx if accepted else xx[:, :len(FINANCIAL)]) @ theta[1:]
    low, high = np.quantile(z, [0.01, 0.99])
    report = dict(data_source="Synthetic outcomes only; not validated on real loans", recovery_selected=bool(accepted),
                  selection_rule="Genius is financial-only; Grit is separate. Combined model results are exploratory only",
                  scores_separate=True, combined_candidate_passed_gate=candidate_passed,
                  financial_validation=base["validation"], combined_validation=combined["validation"],
                  chosen_penalty=selected["penalty"], train_observations=len(data["train"][1]),
                  train_last_feature_month=47, train_label_boundary=59, validation_last_feature_month=83,
                  validation_label_boundary=95, test_first_feature_month=96, test_last_feature_month=T - 13)
    model = CreditModel(fills, theta, bool(accepted), float(low), float(high), report)
    tx, ty = data["test"]
    for label, candidate in (("financial_test", base), ("combined_test", combined)):
        xx = tx if candidate["recovery"] else tx[:, :len(FINANCIAL)]
        report[label] = metrics(ty, expit(candidate["theta"][0] + xx @ candidate["theta"][1:]))
    report["weights"] = model.weights()
    log(f"Outcome-trained Genius: recovery {'included' if accepted else 'not selected'}; "
        f"validation AUC financial={base['validation']['auc']:.3f}, combined={combined['validation']['auc']:.3f}")
    return model
