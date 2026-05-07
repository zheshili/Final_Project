from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, root, minimize_scalar
from scipy.stats import chi2




# --------------------------------------------------------------
# 0. Project paths
# --------------------------------------------------------------
PROJECT_DIR = Path(r"D:\李哲实文档\Stony Brook\AMS561\Project\Final_Project")
RUST_DATA_REPO = PROJECT_DIR / "rust-data"
DATA_DIR = RUST_DATA_REPO / "data"
OUTDIR = PROJECT_DIR / "rust_replication_outputs_simplified"

print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("PROJECT_DIR:", PROJECT_DIR)
print("RUST_DATA_REPO:", RUST_DATA_REPO)
print("DATA_DIR:", DATA_DIR)
print("OUTDIR:", OUTDIR)
print("DATA_DIR exists:", DATA_DIR.exists())
print("data_processing.py exists:", (DATA_DIR / "data_processing.py").exists())

if not PROJECT_DIR.exists():
    raise FileNotFoundError(f"PROJECT_DIR does not exist: {PROJECT_DIR}")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")
if not (DATA_DIR / "data_processing.py").exists():
    raise FileNotFoundError(f"Cannot find data_processing.py in: {DATA_DIR}")

OUTDIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))
from data_processing import data_reading, data_processing  # noqa: E402


# --------------------------------------------------------------
# 1. Model specification
# --------------------------------------------------------------
@dataclass
class RustNFXPSpec:
    num_states: int = 90
    binsize: int = 5000
    beta: float = 0.9999

    # inner solver controls
    fp_tol: float = 1e-8
    fp_max_iter: int = 50000
    root_tol: float = 1e-8
    root_maxfev: int = 200000

    # numerical safeguards
    prob_floor: float = 1e-300
    penalty_value: float = 1e12

    # equilibrium / demand controls
    stationary_tol: float = 1e-14
    stationary_max_iter: int = 500000
    demand_grid_points: int = 30
    demand_rc_scale_low: float = 0.50
    demand_rc_scale_high: float = 1.50

    # debugging
    verbose: bool = False
    print_every: int = 5000


_LAST_W: Optional[np.ndarray] = None


# --------------------------------------------------------------
# 2. Data preparation
# --------------------------------------------------------------
def build_panel(groups: Iterable[str], spec: RustNFXPSpec) -> pd.DataFrame:
    """Build a pooled panel from rust-data output."""
    frames = []
    next_bus_id = 0

    for group in groups:
        print(f"\nProcessing {group} ...")
        part = data_processing({"groups": group, "binsize": spec.binsize}, pickle=False).reset_index()

        old_ids = sorted(part["Bus_ID"].astype(int).unique())
        id_map = {old: next_bus_id + i for i, old in enumerate(old_ids)}
        part["Bus_ID"] = part["Bus_ID"].map(id_map).astype(int)
        next_bus_id += len(old_ids)

        part["group"] = group
        part["state"] = pd.to_numeric(part["state"], errors="raise").astype(int)
        part["decision"] = pd.to_numeric(part["decision"], errors="raise").astype(int)
        part["usage_raw"] = pd.to_numeric(part["usage"], errors="coerce")
        part["usage"] = part["usage_raw"].clip(lower=0, upper=2).astype("Int64")
        part["state"] = part["state"].clip(lower=0, upper=spec.num_states - 1)

        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Bus_ID", "period"]).set_index(["Bus_ID", "period"])

    print("\nPanel summary")
    print("  Buses:", df.index.get_level_values("Bus_ID").nunique())
    print("  Observations:", len(df))
    print("  Replace share:", df["decision"].mean())
    print("  Max state:", df["state"].max())
    print("  Usage counts:")
    print(df["usage"].value_counts(dropna=False).sort_index())

    return df



def make_estimation_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    states = df["state"].to_numpy(dtype=int)
    decisions = df["decision"].to_numpy(dtype=int)
    usage = df["usage"].dropna().to_numpy(dtype=int)

    if not set(np.unique(decisions)).issubset({0, 1}):
        raise ValueError("decision must be binary.")
    if not set(np.unique(usage)).issubset({0, 1, 2}):
        raise ValueError("usage must be in {0,1,2} after clipping.")

    return {
        "states": states,
        "decisions": decisions,
        "usage": usage,
    }


# --------------------------------------------------------------
# 3. Parameter transforms and delta-method objects
# --------------------------------------------------------------
def softmax3(z0: float, z1: float) -> np.ndarray:
    z = np.array([z0, z1, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()



def unpack_params(phi: np.ndarray) -> Tuple[float, float, np.ndarray]:
    log_rc, log_theta, z0, z1 = phi
    rc = np.exp(log_rc)
    theta = np.exp(log_theta)
    probs = softmax3(z0, z1)
    return rc, theta, probs



def transform_jacobian(phi: np.ndarray) -> np.ndarray:
    """
    Jacobian of [RC, theta, p0, p1, p2] with respect to
    [log_RC, log_theta, z0, z1].
    """
    rc, theta, probs = unpack_params(phi)
    p0, p1, p2 = probs

    j = np.zeros((5, 4), dtype=float)
    j[0, 0] = rc
    j[1, 1] = theta

    dp_dz0 = np.array([
        p0 * (1.0 - p0),
        -p0 * p1,
        -p0 * p2,
    ])
    dp_dz1 = np.array([
        -p0 * p1,
        p1 * (1.0 - p1),
        -p1 * p2,
    ])

    j[2:, 2] = dp_dz0
    j[2:, 3] = dp_dz1
    return j



def extract_covariance_from_result(result) -> Optional[np.ndarray]:
    """Best-effort extraction of the approximate covariance matrix from L-BFGS-B."""
    try:
        hess_inv = result.hess_inv
        if hasattr(hess_inv, "todense"):
            cov = np.asarray(hess_inv.todense(), dtype=float)
        else:
            cov = np.asarray(hess_inv, dtype=float)
        if cov.shape != (4, 4):
            return None
        if not np.all(np.isfinite(cov)):
            return None
        return cov
    except Exception:
        return None


# --------------------------------------------------------------
# 4. Economic primitives
# --------------------------------------------------------------
def maintenance_cost(num_states: int, theta: float) -> np.ndarray:
    x = np.arange(num_states, dtype=float)
    return theta * x



def build_keep_transition_matrix(num_states: int, probs: np.ndarray) -> np.ndarray:
    p0, p1, p2 = probs
    p = np.zeros((num_states, num_states), dtype=float)

    for x in range(num_states):
        p[x, x] += p0
        p[x, min(x + 1, num_states - 1)] += p1
        p[x, min(x + 2, num_states - 1)] += p2

    return p



def bellman_map(w: np.ndarray, p_keep: np.ndarray, c: np.ndarray, rc: float, beta: float) -> np.ndarray:
    ev_keep = p_keep @ w
    v_keep = -c + beta * ev_keep
    v_rep = -rc - c[0] + beta * ev_keep[0]
    return np.logaddexp(v_keep, v_rep)


# --------------------------------------------------------------
# 5. Inner loop: solve Bellman fixed point
# --------------------------------------------------------------
def solve_ex_ante_value(
    rc: float,
    theta: float,
    probs: np.ndarray,
    spec: RustNFXPSpec,
    w_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_keep = build_keep_transition_matrix(spec.num_states, probs)
    c = maintenance_cost(spec.num_states, theta)

    if w_init is None:
        w = np.logaddexp(-c, -rc - c[0])
    else:
        w = np.asarray(w_init, dtype=float).copy()

    diff = np.inf
    for it in range(spec.fp_max_iter):
        w_new = bellman_map(w, p_keep, c, rc, spec.beta)
        diff = np.max(np.abs(w_new - w))

        if spec.verbose and (it % spec.print_every == 0):
            print(f"  fixed-point iter {it:>7}, max diff = {diff:.6e}")

        if diff < spec.fp_tol:
            return w_new, p_keep, c
        w = w_new

    if spec.verbose:
        print("  value iteration did not converge; switching to root solver.")

    def residual(w_vec: np.ndarray) -> np.ndarray:
        return w_vec - bellman_map(w_vec, p_keep, c, rc, spec.beta)

    sol = root(
        residual,
        w,
        method="hybr",
        options={"xtol": spec.root_tol, "maxfev": spec.root_maxfev},
    )

    if sol.success:
        return np.asarray(sol.x, dtype=float), p_keep, c

    raise RuntimeError(
        f"Fixed point did not converge. Last VI max diff = {diff:.6e}. Root message: {sol.message}"
    )



def choice_probabilities(
    w: np.ndarray,
    p_keep: np.ndarray,
    c: np.ndarray,
    rc: float,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ev_keep = p_keep @ w
    v_keep = -c + beta * ev_keep
    v_rep = -rc - c[0] + beta * ev_keep[0]

    log_denom = np.logaddexp(v_keep, v_rep)
    p_keep_choice = np.exp(v_keep - log_denom)
    p_rep_choice = np.exp(v_rep - log_denom)
    return p_keep_choice, p_rep_choice


# --------------------------------------------------------------
# 6. Equilibrium objects and replacement demand
# --------------------------------------------------------------
def policy_transition_matrix(p_keep_transition: np.ndarray, p_keep_choice: np.ndarray, p_rep_choice: np.ndarray) -> np.ndarray:
    """
    Transition matrix induced by the optimal policy.

    If the engine is kept in state x, the next state follows row x of P_keep.
    If the engine is replaced in state x, next period starts from a new engine,
    so the next-state distribution equals row 0 of P_keep.
    """
    p_policy = p_keep_choice[:, None] * p_keep_transition
    replacement_next = p_keep_transition[0, :]
    p_policy += p_rep_choice[:, None] * replacement_next[None, :]

    row_sums = p_policy.sum(axis=1, keepdims=True)
    p_policy = p_policy / np.clip(row_sums, 1e-300, None)
    return p_policy



def stationary_distribution_from_transition(p: np.ndarray, tol: float = 1e-14, max_iter: int = 500000) -> np.ndarray:
    n = p.shape[0]
    dist = np.zeros(n, dtype=float)
    dist[0] = 1.0

    for _ in range(max_iter):
        new_dist = dist @ p
        if np.max(np.abs(new_dist - dist)) < tol:
            return new_dist / new_dist.sum()
        dist = new_dist

    return dist / dist.sum()



def expected_replacement_rate(stationary_dist: np.ndarray, p_replace: np.ndarray) -> float:
    return float(np.dot(stationary_dist, p_replace))



def expected_mileage_from_stationary(stationary_dist: np.ndarray, binsize: int) -> float:
    states = np.arange(len(stationary_dist), dtype=float)
    return float(np.dot(stationary_dist, states * binsize))



def demand_curve_grid(fit: Dict[str, object], spec: RustNFXPSpec) -> pd.DataFrame:
    """
    Compute the expected replacement demand function by varying RC and, for each
    counterfactual RC, recomputing the value function, policy, stationary mileage
    distribution, and stationary expected replacement rate.
    """
    rc_hat = float(fit["RC"])
    theta = float(fit["theta"])
    probs = np.asarray(fit["transition_probs"], dtype=float)
    rc_grid = np.linspace(spec.demand_rc_scale_low * rc_hat, spec.demand_rc_scale_high * rc_hat, spec.demand_grid_points)

    rows = []
    w_init = np.asarray(fit["W"], dtype=float).copy()

    for rc_cf in rc_grid:
        w_cf, p_keep_cf, c_cf = solve_ex_ante_value(rc_cf, theta, probs, spec, w_init=w_init)
        p_keep_choice_cf, p_replace_cf = choice_probabilities(w_cf, p_keep_cf, c_cf, rc_cf, spec.beta)
        p_policy_cf = policy_transition_matrix(p_keep_cf, p_keep_choice_cf, p_replace_cf)
        stationary_cf = stationary_distribution_from_transition(
            p_policy_cf,
            tol=spec.stationary_tol,
            max_iter=spec.stationary_max_iter,
        )
        repl_rate_cf = expected_replacement_rate(stationary_cf, p_replace_cf)
        mean_miles_cf = expected_mileage_from_stationary(stationary_cf, spec.binsize)

        rows.append(
            {
                "RC": float(rc_cf),
                "RC_over_hat": float(rc_cf / rc_hat),
                "Expected replacement rate": float(repl_rate_cf),
                "Expected mileage": float(mean_miles_cf),
            }
        )
        w_init = w_cf.copy()

    return pd.DataFrame(rows)


# --------------------------------------------------------------
# 7. Log-likelihood
# --------------------------------------------------------------
def negative_log_likelihood(phi: np.ndarray, data: Dict[str, np.ndarray], spec: RustNFXPSpec) -> float:
    global _LAST_W

    rc, theta, probs = unpack_params(phi)
    if (not np.isfinite(rc)) or (not np.isfinite(theta)) or np.any(~np.isfinite(probs)):
        return spec.penalty_value

    try:
        w, p_keep, c = solve_ex_ante_value(rc, theta, probs, spec, w_init=_LAST_W)
        _LAST_W = w.copy()
    except RuntimeError:
        return spec.penalty_value

    p_keep_choice, p_rep_choice = choice_probabilities(w, p_keep, c, rc, spec.beta)

    states = data["states"]
    decisions = data["decisions"]
    chosen_probs = np.where(decisions == 1, p_rep_choice[states], p_keep_choice[states])
    chosen_probs = np.clip(chosen_probs, spec.prob_floor, 1.0)
    ll_choice = np.log(chosen_probs).sum()

    usage = data["usage"]
    counts = np.bincount(usage, minlength=3)
    probs_safe = np.clip(probs, spec.prob_floor, 1.0)
    ll_transition = (counts * np.log(probs_safe)).sum()

    nll = -(ll_choice + ll_transition)
    if not np.isfinite(nll):
        return spec.penalty_value
    return nll


# --------------------------------------------------------------
# 8. Estimation
# --------------------------------------------------------------
def initial_guess(data: Dict[str, np.ndarray]) -> np.ndarray:
    usage = data["usage"]
    shares = np.bincount(usage, minlength=3).astype(float)
    shares /= shares.sum()

    eps = 1e-8
    shares = np.clip(shares, eps, 1.0)
    shares /= shares.sum()

    z0 = np.log(shares[0] / shares[2])
    z1 = np.log(shares[1] / shares[2])
    log_rc = np.log(8.0)
    log_theta = np.log(0.25)

    return np.array([log_rc, log_theta, z0, z1], dtype=float)



def estimate_model(df: pd.DataFrame, spec: RustNFXPSpec) -> Dict[str, object]:
    global _LAST_W
    _LAST_W = None

    data = make_estimation_data(df)
    x0 = initial_guess(data)

    rc0, theta0, probs0 = unpack_params(x0)
    print("\nInitial guess")
    print("  RC =", rc0)
    print("  theta =", theta0)
    print("  probs =", probs0)

    w0, _, _ = solve_ex_ante_value(rc0, theta0, probs0, spec, w_init=None)
    _LAST_W = w0.copy()
    print("  Initial fixed point solved successfully.")

    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]

    result = minimize(
        fun=negative_log_likelihood,
        x0=x0,
        args=(data, spec),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "disp": True},
    )

    rc, theta, probs = unpack_params(result.x)
    w, p_keep_transition, c = solve_ex_ante_value(rc, theta, probs, spec, w_init=_LAST_W)
    p_keep_choice, p_rep_choice = choice_probabilities(w, p_keep_transition, c, rc, spec.beta)

    p_policy = policy_transition_matrix(p_keep_transition, p_keep_choice, p_rep_choice)
    stationary = stationary_distribution_from_transition(
        p_policy,
        tol=spec.stationary_tol,
        max_iter=spec.stationary_max_iter,
    )
    repl_rate = expected_replacement_rate(stationary, p_rep_choice)
    mean_miles = expected_mileage_from_stationary(stationary, spec.binsize)

    cov_phi = extract_covariance_from_result(result)
    se_transformed = None
    if cov_phi is not None:
        j = transform_jacobian(result.x)
        cov_transformed = j @ cov_phi @ j.T
        diag = np.diag(cov_transformed).copy()
        diag[diag < 0.0] = np.nan
        se_transformed = np.sqrt(diag)

    demand_grid = demand_curve_grid(
        {
            "RC": float(rc),
            "theta": float(theta),
            "transition_probs": np.asarray(probs, dtype=float),
            "W": np.asarray(w, dtype=float),
        },
        spec,
    )

    return {
        "optimizer_result": result,
        "RC": float(rc),
        "theta": float(theta),
        "transition_probs": np.asarray(probs, dtype=float),
        "W": np.asarray(w, dtype=float),
        "p_keep": np.asarray(p_keep_choice, dtype=float),
        "p_replace": np.asarray(p_rep_choice, dtype=float),
        "P_keep": np.asarray(p_keep_transition, dtype=float),
        "P_policy": np.asarray(p_policy, dtype=float),
        "stationary_distribution": np.asarray(stationary, dtype=float),
        "expected_replacement_rate": float(repl_rate),
        "expected_stationary_mileage": float(mean_miles),
        "demand_grid": demand_grid,
        "c": np.asarray(c, dtype=float),
        "cov_phi": cov_phi,
        "se_transformed": se_transformed,
        "spec": spec,
        "data": data,
        "n_obs": int(len(df)),
        "n_buses": int(df.index.get_level_values("Bus_ID").nunique()),
        "replace_share": float(df["decision"].mean()),
    }


# --------------------------------------------------------------
# 9. Tests and summaries
# --------------------------------------------------------------
def empirical_hazard(df: pd.DataFrame, num_states: int) -> pd.DataFrame:
    d = df.reset_index().copy()
    grouped = d.groupby("state")
    obs = grouped["decision"].size().reindex(range(num_states), fill_value=0)
    repl = grouped["decision"].sum().reindex(range(num_states), fill_value=0)
    hazard = repl / obs.replace(0, np.nan)

    return pd.DataFrame(
        {
            "state": np.arange(num_states),
            "n_obs": obs.values,
            "n_replace": repl.values,
            "hazard": hazard.values,
        }
    )



def lm_independence_test(df: pd.DataFrame, fit: Dict[str, object]) -> Dict[str, float]:
    """
    Simple LM-style specification test motivated by Rust's Table XI.

    Null: conditional on current state, lagged replacement does not affect the
    current choice probability.

    We use the fitted structural probability as a fixed offset and test whether the
    coefficient on lagged decision is zero in a one-parameter logit augmentation:

        logit P(i_t = 1 | x_t, i_{t-1}) = logit(p_hat(x_t)) + a * i_{t-1}

    The resulting score statistic is asymptotically chi-square with 1 degree of
    freedom. This is a compact implementation.
    """
    d = df.reset_index().copy().sort_values(["Bus_ID", "period"]).reset_index(drop=True)
    d["lag_decision"] = d.groupby("Bus_ID")["decision"].shift(1)
    d = d.dropna(subset=["lag_decision"]).copy()

    y = d["decision"].to_numpy(dtype=float)
    lag = d["lag_decision"].to_numpy(dtype=float)
    states = d["state"].to_numpy(dtype=int)
    p = np.clip(np.asarray(fit["p_replace"], dtype=float)[states], 1e-10, 1.0 - 1e-10)

    score = np.sum(lag * (y - p))
    info = np.sum((lag ** 2) * p * (1.0 - p))

    if info <= 0.0 or not np.isfinite(info):
        lm_stat = np.nan
        pval = np.nan
        a_hat = np.nan
    else:
        lm_stat = float((score ** 2) / info)
        pval = float(1.0 - chi2.cdf(lm_stat, df=1))

        offset = np.log(p / (1.0 - p))

        def neg_augmented_ll(a: float) -> float:
            eta = offset + a * lag
            p_alt = 1.0 / (1.0 + np.exp(-eta))
            p_alt = np.clip(p_alt, 1e-12, 1.0 - 1e-12)
            return -np.sum(y * np.log(p_alt) + (1.0 - y) * np.log(1.0 - p_alt))

        try:
            opt = minimize_scalar(neg_augmented_ll, bounds=(-10.0, 10.0), method="bounded")
            a_hat = float(opt.x) if opt.success else np.nan
        except Exception:
            a_hat = np.nan

    return {
        "lm_stat": lm_stat,
        "pvalue": pval,
        "df": 1.0,
        "n_test_obs": float(len(d)),
        "a_hat": a_hat,
    }



def heterogeneity_lr_test(fit_123: Dict[str, object], fit_4: Dict[str, object], fit_1234: Dict[str, object]) -> Dict[str, float]:
    ll_sep = -float(fit_123["optimizer_result"].fun) + -float(fit_4["optimizer_result"].fun)
    ll_pool = -float(fit_1234["optimizer_result"].fun)
    stat = 2.0 * (ll_sep - ll_pool)
    df = 4.0
    pval = 1.0 - chi2.cdf(max(stat, 0.0), df=int(df))
    return {
        "lr_stat": float(stat),
        "df": df,
        "pvalue": float(pval),
        "ll_separate": float(ll_sep),
        "ll_pooled": float(ll_pool),
    }


# --------------------------------------------------------------
# 10. Table helpers
# --------------------------------------------------------------
def sanitize_name(name: str) -> str:
    return name.replace(",", "").replace(" ", "_").replace("-", "_")



def latex_escape(text: object) -> str:
    s = str(text)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s



def format_float(x: object, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)



def est_se_string(est: float, se: float | None, digits: int = 4) -> str:
    if se is None or pd.isna(se):
        return format_float(est, digits)
    return f"{format_float(est, digits)} ({format_float(se, digits)})"



def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{'l' + 'c' * (len(cols) - 1)}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            val = row[c]
            vals.append(val if isinstance(val, str) else latex_escape(format_float(val)))
        lines.append(" & ".join(vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")



def fit_to_estimation_row(sample_name: str, fit: Dict[str, object]) -> Dict[str, object]:
    se = fit.get("se_transformed")
    se_rc = None if se is None else float(se[0])
    se_theta = None if se is None else float(se[1])
    se_p0 = None if se is None else float(se[2])
    se_p1 = None if se is None else float(se[3])
    se_p2 = None if se is None else float(se[4])

    p0, p1, p2 = [float(x) for x in fit["transition_probs"]]

    return {
        "Sample": sample_name,
        "RC": est_se_string(float(fit["RC"]), se_rc),
        "theta": est_se_string(float(fit["theta"]), se_theta),
        "p0": est_se_string(p0, se_p0),
        "p1": est_se_string(p1, se_p1),
        "p2": est_se_string(p2, se_p2),
        "Log likelihood": format_float(-float(fit["optimizer_result"].fun), 3),
        "N obs": str(int(fit["n_obs"])),
        "N buses": str(int(fit["n_buses"])),
    }



def equilibrium_summary_row(sample_name: str, fit: Dict[str, object]) -> Dict[str, object]:
    stationary = np.asarray(fit["stationary_distribution"], dtype=float)
    states = np.arange(len(stationary), dtype=float)
    mean_state = float(np.dot(stationary, states))
    second_moment = float(np.dot(stationary, states ** 2))
    sd_state = float(np.sqrt(max(second_moment - mean_state ** 2, 0.0)))
    cum = np.cumsum(stationary)
    median_state = int(np.searchsorted(cum, 0.5))

    return {
        "Sample": sample_name,
        "Mean mileage": format_float(fit["expected_stationary_mileage"], 1),
        "Mean state": format_float(mean_state, 3),
        "SD state": format_float(sd_state, 3),
        "Median state": str(median_state),
        "Expected replacement rate": format_float(fit["expected_replacement_rate"], 6),
    }



def demand_grid_to_table(sample_name: str, demand_grid: pd.DataFrame, n_points: int = 7) -> pd.DataFrame:
    idx = np.linspace(0, len(demand_grid) - 1, n_points).round().astype(int)
    idx = np.unique(idx)
    sub = demand_grid.iloc[idx].copy().reset_index(drop=True)
    sub.insert(0, "Sample", [sample_name] + [""] * (len(sub) - 1))

    out = pd.DataFrame(
        {
            "Sample": sub["Sample"],
            "RC": sub["RC"].map(lambda x: format_float(x, 4)),
            "RC / RC_hat": sub["RC_over_hat"].map(lambda x: format_float(x, 3)),
            "Expected replacement rate": sub["Expected replacement rate"].map(lambda x: format_float(x, 6)),
            "Expected mileage": sub["Expected mileage"].map(lambda x: format_float(x, 1)),
        }
    )
    return out



def lm_test_row(sample_name: str, test: Dict[str, float]) -> Dict[str, object]:
    return {
        "Sample": sample_name,
        "LM statistic": format_float(test["lm_stat"], 4),
        "df": str(int(test["df"])),
        "p-value": format_float(test["pvalue"], 6),
        "a_hat": format_float(test["a_hat"], 4),
        "N test obs": str(int(test["n_test_obs"])),
    }


# --------------------------------------------------------------
# 11. Figures
# --------------------------------------------------------------
def state_to_miles(states: np.ndarray, binsize: int) -> np.ndarray:
    return np.asarray(states) * binsize



def plot_value_function(fit: Dict[str, object], savepath: Path) -> None:
    binsize = fit["spec"].binsize
    x = state_to_miles(np.arange(fit["spec"].num_states), binsize) / 1000.0

    plt.figure(figsize=(9, 6))
    plt.plot(x, fit["W"])
    plt.xlabel("Mileage state (thousands)")
    plt.ylabel("Ex-ante value W(x)")
    plt.title("Estimated value function")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()



def plot_hazard_function(fit: Dict[str, object], empirical_hz: pd.DataFrame, savepath: Path) -> None:
    binsize = fit["spec"].binsize
    x = state_to_miles(np.arange(fit["spec"].num_states), binsize) / 1000.0

    plt.figure(figsize=(9, 6))
    plt.plot(x, fit["p_replace"], label="model hazard")
    plt.scatter(
        empirical_hz["state"] * binsize / 1000.0,
        empirical_hz["hazard"],
        s=18,
        alpha=0.7,
        label="empirical hazard",
    )
    plt.xlabel("Mileage state (thousands)")
    plt.ylabel("Replacement hazard")
    plt.title("Estimated replacement hazard")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()



def plot_equilibrium_distribution(fit: Dict[str, object], savepath: Path) -> None:
    binsize = fit["spec"].binsize
    x = state_to_miles(np.arange(fit["spec"].num_states), binsize) / 1000.0
    stationary = np.asarray(fit["stationary_distribution"], dtype=float)

    plt.figure(figsize=(9, 6))
    plt.plot(x, stationary)
    plt.xlabel("Mileage state (thousands)")
    plt.ylabel("Stationary probability")
    plt.title("Equilibrium distribution of bus mileage")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()



def plot_expected_replacement_demand_function(fit: Dict[str, object], savepath: Path) -> None:
    demand = fit["demand_grid"].copy()

    plt.figure(figsize=(9, 6))
    plt.plot(demand["RC"], demand["Expected replacement rate"])
    plt.axvline(float(fit["RC"]), linestyle="--", linewidth=1.0)
    plt.xlabel("Replacement cost RC")
    plt.ylabel("Expected replacement rate")
    plt.title("Expected replacement demand function")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()



def save_sample_figures(sample_name: str, df: pd.DataFrame, fit: Dict[str, object], outdir: Path) -> None:
    sample_dir = outdir / sanitize_name(sample_name)
    sample_dir.mkdir(parents=True, exist_ok=True)

    empirical_hz = empirical_hazard(df, fit["spec"].num_states)
    plot_value_function(fit, sample_dir / "value_function.png")
    plot_hazard_function(fit, empirical_hz, sample_dir / "hazard_function.png")
    plot_equilibrium_distribution(fit, sample_dir / "equilibrium_distribution.png")
    plot_expected_replacement_demand_function(fit, sample_dir / "expected_replacement_demand_function.png")


# --------------------------------------------------------------
# 12. Main procedure
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\nStarting data_reading() ...")
    data_reading()
    print("data_reading() finished.\n")

    spec = RustNFXPSpec(
        num_states=90,
        binsize=5000,
        beta=0.9999,
        fp_tol=1e-8,
        fp_max_iter=50000,
        root_tol=1e-8,
        root_maxfev=200000,
        stationary_tol=1e-14,
        stationary_max_iter=500000,
        demand_grid_points=30,
        demand_rc_scale_low=0.50,
        demand_rc_scale_high=1.50,
        verbose=False,
    )

    # paper-style samples
    df_123 = build_panel(["group_1", "group_2", "group_3"], spec)
    df_4 = build_panel(["group_4"], spec)
    df_1234 = build_panel(["group_1", "group_2", "group_3", "group_4"], spec)

    fit_123 = estimate_model(df_123, spec)
    fit_4 = estimate_model(df_4, spec)
    fit_1234 = estimate_model(df_1234, spec)

    # parameter table
    estimation_table = pd.DataFrame(
        [
            fit_to_estimation_row("Groups 1,2,3", fit_123),
            fit_to_estimation_row("Group 4", fit_4),
            fit_to_estimation_row("Groups 1-4 pooled", fit_1234),
        ]
    )
    write_latex_table(
        estimation_table,
        OUTDIR / "estimation_results.tex",
        caption="Dynamic NFXP estimation results for the linear-cost specification.",
        label="tab:rust_simplified_estimation",
    )

    # LM-style independence test table
    lm_123 = lm_independence_test(df_123, fit_123)
    lm_4 = lm_independence_test(df_4, fit_4)
    lm_1234 = lm_independence_test(df_1234, fit_1234)
    lm_table = pd.DataFrame(
        [
            lm_test_row("Groups 1,2,3", lm_123),
            lm_test_row("Group 4", lm_4),
            lm_test_row("Groups 1-4 pooled", lm_1234),
        ]
    )
    write_latex_table(
        lm_table,
        OUTDIR / "lm_specification_test.tex",
        caption="LM-style specification test for lagged-decision dependence.",
        label="tab:rust_simplified_lm_test",
    )

    # heterogeneity LR test table
    heterogeneity = heterogeneity_lr_test(fit_123, fit_4, fit_1234)
    heterogeneity_table = pd.DataFrame(
        [
            {
                "Restricted model": "Groups 1-4 pooled",
                "Unrestricted model": "Groups 1,2,3 + Group 4",
                "LR statistic": format_float(heterogeneity["lr_stat"], 4),
                "df": str(int(heterogeneity["df"])),
                "p-value": format_float(heterogeneity["pvalue"], 6),
                "LL restricted": format_float(heterogeneity["ll_pooled"], 3),
                "LL unrestricted": format_float(heterogeneity["ll_separate"], 3),
            }
        ]
    )
    write_latex_table(
        heterogeneity_table,
        OUTDIR / "heterogeneity_lr_test.tex",
        caption="Likelihood-ratio heterogeneity test across the paper-style samples.",
        label="tab:rust_simplified_heterogeneity",
    )

    # equilibrium distribution summary table
    equilibrium_summary_table = pd.DataFrame(
        [
            equilibrium_summary_row("Groups 1,2,3", fit_123),
            equilibrium_summary_row("Group 4", fit_4),
            equilibrium_summary_row("Groups 1-4 pooled", fit_1234),
        ]
    )
    write_latex_table(
        equilibrium_summary_table,
        OUTDIR / "equilibrium_distribution_summary.tex",
        caption="Summary moments of the equilibrium distribution of bus mileage.",
        label="tab:rust_equilibrium_distribution_summary",
    )

    # expected replacement demand grid table
    demand_table = pd.concat(
        [
            demand_grid_to_table("Groups 1,2,3", fit_123["demand_grid"]),
            demand_grid_to_table("Group 4", fit_4["demand_grid"]),
            demand_grid_to_table("Groups 1-4 pooled", fit_1234["demand_grid"]),
        ],
        ignore_index=True,
    )
    write_latex_table(
        demand_table,
        OUTDIR / "expected_replacement_demand_grid.tex",
        caption="Selected grid points from the expected replacement demand function.",
        label="tab:rust_expected_replacement_demand_grid",
    )

    # figures
    save_sample_figures("Groups 1,2,3", df_123, fit_123, OUTDIR)
    save_sample_figures("Group 4", df_4, fit_4, OUTDIR)
    save_sample_figures("Groups 1-4 pooled", df_1234, fit_1234, OUTDIR)

    print("\nAll simplified outputs written to:", OUTDIR.resolve())
    print("Files created:")
    print("  -", OUTDIR / "estimation_results.tex")
    print("  -", OUTDIR / "lm_specification_test.tex")
    print("  -", OUTDIR / "heterogeneity_lr_test.tex")
    print("  -", OUTDIR / "equilibrium_distribution_summary.tex")
    print("  -", OUTDIR / "expected_replacement_demand_grid.tex")
    print("  - per-sample folders with:")
    print("      value_function.png")
    print("      hazard_function.png")
    print("      equilibrium_distribution.png")
    print("      expected_replacement_demand_function.png")
