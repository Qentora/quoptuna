"""Freeze a model per search arm, set the offer budget, and audit it.

Selection rules (fixed before the final audit run):
  off          the trial with the best validation F1.
  constrained  the best validation F1 among feasible trials (equal-opportunity
               gap <= 0.10) that find at least one owner (F1 > 0); an arm with
               no such trial reports "none".
  pareto       Pareto front of (val F1 up, gap down) without F1 = 0 trials;
               among points within 90% of the front's best F1, the lowest gap.

Deployment rule: the program targets a budget of 15% of households, the top of
the score ranking. The score cutoff is set on the calibration set
(survey-weighted) and applied unchanged to the audit set, so the realised
audit share ("invited_share") is close to, not exactly, 15%.

Audit metrics (audit set, survey-weighted with NWEIGHT): owners invited per
income group (true-positive rate), equal-opportunity gap (max - min TPR),
invitation yield (share of invitees that own one), share of invitations per
group; 95% intervals from 2,000 stratified bootstrap resamples. Scoring cost
per household with library thread limits of two (run_all.sh).

Mitigation (release-gate fix, same 15% target): two cutoffs set on the
calibration set, one for lower-income households (<$35k) and one for everyone
else, so both groups' owners are invited at about the same rate. Two groups,
not three, because a utility can plausibly know only a lower-income flag (from
income-qualified rate or assistance enrollment). Reported under "post" for
every model and still audited across all three income bands. "post_bands" is
the upper bound with all three bands known at decision time. The audit set was
scored once before this step existed (pipeline test on a stopped run); the
mitigation results are therefore exploratory.
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from prepare_data import DATA, RESULTS_DIR
from run_search import RESULTS, SENSITIVE, encode, fit_scaler, pin_mlp_seed
from sklearn.linear_model import LogisticRegression

BUDGET = 0.15
EO_TOLERANCE = 0.10
NEAR_BEST = 0.90
N_BOOT = 2000
GROUPS = ["lt35k", "35to100k", "ge100k"]
GROUP_LABEL = {"lt35k": "<$35k", "35to100k": "$35k-$100k", "ge100k": ">=$100k"}
VARIATIONAL = {  # JAX-trained families: the only ones ASHA can prune
    "CircuitCentricClassifier",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierSeparable",
    "QuantumMetricLearner",
    "TreeTensorClassifier",
    "SeparableVariationalClassifier",
}
CLASSICAL = {"SVC", "SVClinear", "MLPClassifier", "Perceptron"}


# ---------------------------------------------------------------- search side
def load_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS.glob("thermostat_*_s*.json")):
        run = json.loads(path.read_text())
        if run["meta"].get("tag"):
            continue
        runs.append(run)
    return runs


def complete(run: dict) -> list[dict]:
    return [t for t in run["trials"] if t["state"] == "COMPLETE"]


def val_f1(t: dict) -> float:
    return float(t["values"][0])


def gap(t: dict) -> float | None:
    v = t["user_attrs"].get("fairness_disparity")
    return None if v is None else float(v)


def duplicates(trials: list[dict]) -> list[dict]:
    seen, out = set(), []
    for t in sorted(trials, key=lambda t: t["number"]):
        k = json.dumps(t["params"], sort_keys=True)
        if k in seen:
            out.append(t)
        seen.add(k)
    return out


def search_cost(run: dict) -> dict:
    trials = run["trials"]
    fam = pd.Series([t["params"].get("model_type") for t in trials]).value_counts().to_dict()
    var_hours = (
        sum(
            t["user_attrs"].get("training_time", 0.0) or 0.0
            for t in trials
            if t["params"].get("model_type") in VARIATIONAL
        )
        / 3600
    )
    var_steps = [
        t["user_attrs"]["n_steps"]
        for t in trials
        if t["params"].get("model_type") in VARIATIONAL and t["user_attrs"].get("n_steps")
    ]
    done = complete(run)
    out = {
        "arm": run["meta"]["arm"],
        "seed": run["meta"]["seed"],
        "trials": len(trials),
        "complete": len(done),
        "pruned": sum(t["state"] == "PRUNED" for t in trials),
        "failed": sum(t["state"] == "FAIL" for t in trials),
        "wall_clock_h": run["meta"]["wall_clock_s"] / 3600,
        "variational_trials": sum(fam.get(m, 0) for m in VARIATIONAL),
        "variational_training_h": var_hours,
        "mean_variational_steps": float(np.mean(var_steps)) if var_steps else None,
        "quantum_metric_learner_trials": fam.get("QuantumMetricLearner", 0),
        "longest_trial_min": max(t["duration_s"] or 0 for t in trials) / 60,
        "longest_trial_family": max(trials, key=lambda t: t["duration_s"] or 0)["params"].get(
            "model_type"
        ),
        # Trials whose configuration repeats an earlier trial of the same run:
        # the sampler re-proposed it and the search paid for it again.
        "duplicate_trials": len(dups := duplicates(trials)),
        "duplicate_min": sum(t["duration_s"] or 0 for t in dups) / 60,
        "qml_h": sum(
            t["duration_s"] or 0
            for t in trials
            if t["params"].get("model_type") == "QuantumMetricLearner"
        )
        / 3600,
        "best_val_f1": max(val_f1(t) for t in done),
        "best_quantum_val_f1": max(
            (val_f1(t) for t in done if t["params"]["model_type"] not in CLASSICAL), default=None
        ),
        "families": fam,
        "family_hours": {
            m: sum(
                t["user_attrs"].get("training_time", 0.0) or 0.0
                for t in trials
                if t["params"].get("model_type") == m
            )
            / 3600
            for m in VARIATIONAL
            if fam.get(m)
        },
    }
    if run["meta"]["arm"] != "off":
        scored = [t for t in done if gap(t) is not None]
        out["degenerate"] = sum(val_f1(t) == 0 for t in scored)
        out["feasible"] = sum(gap(t) <= EO_TOLERANCE for t in scored)
        out["feasible_nondegenerate"] = sum(
            gap(t) <= EO_TOLERANCE and val_f1(t) > 0 for t in scored
        )
    return out


def pareto_front(trials: list[dict]) -> list[dict]:
    pts = [t for t in trials if val_f1(t) > 0 and gap(t) is not None]
    return [
        t
        for t in pts
        if not any(
            val_f1(o) >= val_f1(t)
            and gap(o) <= gap(t)
            and (val_f1(o) > val_f1(t) or gap(o) < gap(t))
            for o in pts
        )
    ]


def select(run: dict) -> dict | None:
    done = complete(run)
    arm = run["meta"]["arm"]
    if arm == "off":
        return max(done, key=val_f1)
    if arm == "constrained":
        ok = [t for t in done if val_f1(t) > 0 and gap(t) is not None and gap(t) <= EO_TOLERANCE]
        return max(ok, key=val_f1) if ok else None
    front = pareto_front(done)
    best = max(val_f1(t) for t in front)
    near = [t for t in front if val_f1(t) >= NEAR_BEST * best]
    return min(near, key=lambda t: (gap(t), -val_f1(t)))


# ---------------------------------------------------------------- audit side
def scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(x))[:, 1]
        except Exception:  # noqa: BLE001, S110 - fall through to decision_function
            pass
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x)).ravel()
    return np.asarray(model.predict(x)).ravel().astype(float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cum, q)])


def program_metrics(invite: np.ndarray, own: np.ndarray, grp: np.ndarray, w: np.ndarray) -> dict:
    out = {"tpr": {}, "offer_share": {}}
    total_offers = (w * invite).sum()
    for g in GROUPS:
        m = grp == g
        owners = (w * own * m).sum()
        out["tpr"][g] = float((w * own * invite * m).sum() / owners) if owners else float("nan")
        out["offer_share"][g] = (
            float((w * invite * m).sum() / total_offers) if total_offers else 0.0
        )
    out["eo_gap"] = float(max(out["tpr"].values()) - min(out["tpr"].values()))
    out["yield"] = float((w * own * invite).sum() / total_offers) if total_offers else float("nan")
    out["recall"] = float((w * own * invite).sum() / (w * own).sum())
    out["invited_share"] = float(total_offers / w.sum())
    return out


def bootstrap(invite, own, grp, w, rng) -> dict:
    strata = pd.Series(grp).astype(str) + "|" + pd.Series(own).astype(str)
    idx_by = [np.flatnonzero(strata.to_numpy() == s) for s in strata.unique()]
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by])
        draws.append(program_metrics(invite[idx], own[idx], grp[idx], w[idx]))

    def ci(get):
        vals = np.array([get(d) for d in draws])
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

    return {
        "tpr": {g: ci(lambda d, g=g: d["tpr"][g]) for g in GROUPS},
        "eo_gap": ci(lambda d: d["eo_gap"]),
        "yield": ci(lambda d: d["yield"]),
    }


def lower_income_flag(grp: np.ndarray) -> np.ndarray:
    """The only group label the mitigation may use at decision time."""
    return np.where(grp == "lt35k", "lower", "other")


def equal_opportunity_cutoffs(sc, own, flag, w) -> dict:
    """One score cutoff per flag value so that about the same share t of owners
    is invited in each, with t chosen so that about BUDGET of all households are
    invited (bisection on t; weighted empirical quantiles, so equality is
    approximate).

    Post-processing in the spirit of Hardt et al.'s equal-opportunity
    thresholds (and Fairlearn's ThresholdOptimizer), constrained to the
    program's offer budget. Set on the calibration set only.
    """
    labels = np.unique(flag)

    def cutoffs(t: float) -> dict:
        out = {}
        for g in labels:
            m = (flag == g) & own
            out[g] = weighted_quantile(sc[m], w[m], min(max(1 - t, 0.0), 0.999999))
        return out

    def invited(cut: dict) -> float:
        thr = np.array([cut[g] for g in flag])
        return float((w * (sc >= thr)).sum() / w.sum())

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if invited(cutoffs(mid)) > BUDGET:
            hi = mid
        else:
            lo = mid
    return cutoffs(lo)


def evaluate(name: str, model, scaler, cal: pd.DataFrame, aud: pd.DataFrame, rng) -> dict:
    xc, yc = encode(cal, scaler)
    t0 = time.perf_counter()
    sc = scores(model, xc)
    per_household_ms = (time.perf_counter() - t0) / len(xc) * 1e3
    wc = cal["nweight"].to_numpy(float)
    cutoff = weighted_quantile(sc, wc, 1 - BUDGET)
    group_cut = equal_opportunity_cutoffs(
        sc, yc == 1, lower_income_flag(cal[SENSITIVE].to_numpy()), wc
    )
    # Upper bound: the same step with all three income bands known at decision time.
    band_cut = equal_opportunity_cutoffs(sc, yc == 1, cal[SENSITIVE].to_numpy(), wc)
    xa, ya = encode(aud, scaler)
    sa = scores(model, xa)
    own = ya == 1
    grp = aud[SENSITIVE].to_numpy()
    w = aud["nweight"].to_numpy(float)
    invite = sa >= cutoff
    res = {"model": name, **program_metrics(invite, own, grp, w)}
    res["ci"] = bootstrap(invite, own, grp, w, rng)
    # Same 15% target, one cutoff for the lower-income flag and one for everyone
    # else, set on calibration (the mitigation step); audited across all bands.
    invite_eo = sa >= np.array([group_cut[g] for g in lower_income_flag(grp)])
    res["post"] = {
        **program_metrics(invite_eo, own, grp, w),
        "ci": bootstrap(invite_eo, own, grp, w, rng),
    }
    invite_bands = sa >= np.array([band_cut[g] for g in grp])
    res["post_bands"] = {
        **program_metrics(invite_bands, own, grp, w),
        "ci": bootstrap(invite_bands, own, grp, w, rng),
    }
    # The same model at its own default cutoff: the operating point the search scored.
    default_pred = np.asarray(model.predict(xa)).ravel() == 1
    res["default_cutoff"] = program_metrics(default_pred, own, grp, w)
    res["score_ms_per_household"] = per_household_ms
    try:
        res["footprint_kb"] = len(pickle.dumps(model)) / 1024
    except Exception:  # noqa: BLE001
        res["footprint_kb"] = None
    return res


def refit(trial: dict, train_x, train_y, max_steps: int, conv: int):
    pin_mlp_seed()  # same MLP random_state as during search
    from quoptuna.backend.models import create_model  # noqa: PLC0415

    params = {k: v for k, v in trial["params"].items() if k != "model_type"}
    model = create_model(
        trial["params"]["model_type"], max_steps=max_steps, convergence_interval=conv, **params
    )
    t0 = time.perf_counter()
    model.fit(train_x, train_y)
    return model, time.perf_counter() - t0


def access_gap() -> dict:
    """Survey-weighted connected-thermostat ownership over the whole RECS file."""
    raw = pd.read_csv(
        DATA / "recs2020_public_v7.csv", usecols=["TYPETHERM", "KOWNRENT", "MONEYPY", "NWEIGHT"]
    )
    raw = raw[raw["TYPETHERM"] >= 0]
    own = (raw["TYPETHERM"] == 3).astype(float)
    w = raw["NWEIGHT"]

    def rate(mask):
        return float((own[mask] * w[mask]).sum() / w[mask].sum())

    inc = pd.cut(raw["MONEYPY"], [0, 9, 14, 16], labels=GROUPS).astype(str)
    return {
        "n_households": len(raw),
        "n_owners": int(own.sum()),
        "overall": rate(pd.Series(True, index=raw.index)),
        "by_income": {g: rate(inc == g) for g in GROUPS},
        "owners_by_income": {g: int(own[inc == g].sum()) for g in GROUPS},
        "renters": rate(raw["KOWNRENT"] == 2),
        "homeowners": rate(raw["KOWNRENT"] == 1),
    }


def main() -> None:
    rng = np.random.default_rng(7)
    runs = load_runs()
    scaler = fit_scaler()
    s = pd.read_csv(DATA / "search.csv")
    train_x, train_y = encode(s[s["split"] == "train"], scaler)
    cal = pd.read_csv(DATA / "calibration.csv")
    aud = pd.read_csv(DATA / "audit.csv")

    summary = {
        "budget": BUDGET,
        "access_gap": access_gap(),
        "search": [],
        "audit": [],
        "quantum_cost": [],
    }
    for run in runs:
        summary["search"].append(search_cost(run))
        tag = f"{run['meta']['arm']}_s{run['meta']['seed']}"
        trial = select(run)
        if trial is None:
            summary["audit"].append({"model": tag, "selected": None})
        else:
            model, fit_s = refit(
                trial,
                train_x,
                train_y,
                run["meta"]["max_steps"],
                run["meta"]["convergence_interval"],
            )
            res = evaluate(tag, model, scaler, cal, aud, rng)
            res["selected"] = {
                "trial": trial["number"],
                "model_type": trial["params"]["model_type"],
                "params": trial["params"],
                "val_f1": val_f1(trial),
                "val_eo_gap": gap(trial),
                "refit_s": fit_s,
            }
            summary["audit"].append(res)
            print(
                tag,
                trial["params"]["model_type"],
                {k: res[k] for k in ("eo_gap", "yield", "recall")},
            )
        if run["meta"]["arm"] == "off":
            # Scoring cost of the accuracy-only search's best quantum candidate,
            # timed on 100 calibration households (simulated circuits are slow
            # enough that a full audit pass is itself a deployment finding).
            quantum = [t for t in complete(run) if t["params"]["model_type"] not in CLASSICAL]
            if quantum:
                best_q = max(quantum, key=val_f1)
                model, fit_s = refit(
                    best_q,
                    train_x,
                    train_y,
                    run["meta"]["max_steps"],
                    run["meta"]["convergence_interval"],
                )
                xc, _ = encode(cal.iloc[:100], scaler)
                t0 = time.perf_counter()
                scores(model, xc)
                summary["quantum_cost"].append(
                    {
                        "run": tag,
                        "model_type": best_q["params"]["model_type"],
                        "params": best_q["params"],
                        "val_f1": val_f1(best_q),
                        "refit_s": fit_s,
                        "score_ms_per_household": (time.perf_counter() - t0) / len(xc) * 1e3,
                    }
                )

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(train_x, train_y)
    summary["audit"].append(
        {
            **evaluate("logistic_baseline", lr, scaler, cal, aud, rng),
            "selected": {"model_type": "LogisticRegression"},
        }
    )
    # Random invitations at the same 15% budget: every owner has a 15% chance in
    # every group, and yield equals the audit set's weighted ownership rate.
    audit_owner = (aud["smart_thermostat"] == "yes").to_numpy(float)
    audit_w = aud["nweight"].to_numpy(float)
    summary["audit"].append(
        {
            "model": "random_baseline",
            "selected": {"model_type": "random"},
            "tpr": dict.fromkeys(GROUPS, BUDGET),
            "eo_gap": 0.0,
            "yield": float((audit_owner * audit_w).sum() / audit_w.sum()),
            "recall": BUDGET,
            "invited_share": BUDGET,
        }
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["access_gap"], indent=1))


if __name__ == "__main__":
    main()
