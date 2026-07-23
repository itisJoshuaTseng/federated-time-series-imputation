"""
Plot MNAR CA-Fix experiment results.
Generates figures matching the existing academic conference style.
Run from:  federated learning/
  python experiments/plot_mnar_ca_fix.py
  python experiments/plot_mnar_ca_fix.py --with-centralized   # after ceiling runs
"""

import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family":      "Microsoft YaHei",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "errorbar.capsize": 4,
})

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "local":        "#888888",
    "fedavg":       "#4878CF",
    "fedprox":      "#D65F5F",
    "fed_ca":       "#6ACC65",   # old (unused in main plots)
    "fed_ca_fix":   "#E07B39",
    "centralized":  "#9B59B6",
}
LABEL = {
    "local":        "Local Only",
    "fedavg":       "FedAvg",
    "fedprox":      "FedProx",
    "fed_ca":       "Fed-CA (舊)",
    "fed_ca_fix":   "Fed-CA (ours)",
    "centralized":  "Centralized (ceiling)",
}

BASE = Path("logs/saits_mnar")
FIG  = Path("experiments/figures")
FIG.mkdir(exist_ok=True)

SETTINGS = [
    ("S1","quantile",0.3),("S1","quantile",0.5),("S1","quantile",0.7),("S1","logit",0.3),
    ("S4","quantile",0.3),("S4","quantile",0.5),("S4","quantile",0.7),("S4","logit",0.3),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def load(fname):
    p = BASE / fname
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))

def get_stats(results, method):
    rs = [r for r in results if r["method"] == method]
    if not rs:
        return None, None
    vals = [r["mean_mae"] for r in rs]
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

def new_file(s, m, r):
    mm = "q" if m == "quantile" else "l"
    rr = str(r).replace(".", "p")
    return f"cafe_fix_v2_{s}_{mm}_rho{rr}_seeds_0-4.json"

def old_recon_file(s, m, r):
    rr = str(r).replace(".", "p")
    MAP = {
        ("S1","quantile",0.3): "mnar_recon_S1_quantile_20260404_082938.json",
        ("S1","quantile",0.5): "mnar_recon_S1_quantile_rho0p5_20260405_003448.json",
        ("S1","quantile",0.7): "mnar_recon_S1_quantile_rho0p7_20260405_135443.json",
        ("S1","logit",   0.3): "mnar_recon_S1_logit_rho0p3_20260404_111523.json",
        ("S4","quantile",0.3): "mnar_recon_S4_quantile_20260403_233418.json",
        ("S4","quantile",0.5): "mnar_recon_S4_quantile_rho0p5_20260405_071616.json",
        ("S4","quantile",0.7): "mnar_recon_S4_quantile_rho0p7_20260405_202331.json",
        ("S4","logit",   0.3): "mnar_recon_S4_logit_rho0p3_20260404_183827.json",
    }
    return MAP[(s, m, r)]

def old_ca_file(s, m, r):
    mm = "quantile" if m == "quantile" else "logit"
    rr = str(r).replace(".", "p")
    return f"mnar_fed_ca_{s}_{mm}_rho{rr}_seeds_0-4.json"

def ceiling_file(s, m, r):
    mm = "q" if m == "quantile" else "l"
    rr = str(r).replace(".", "p")
    return f"centralized_ceiling_{s}_{mm}_rho{rr}_seeds_0-4.json"

def load_per_seed(s, m, r):
    """Return per-seed MAE lists for {local, fedavg, fedprox, fed_ca_fix}.

    fedprox/fedavg/local come from mnar_recon_*.json (Step 2b baseline run).
    fed_ca_fix comes from cafe_fix_v2_*.json (CA-fix overnight run).
    """
    per_seed = {}
    od = load(old_recon_file(s, m, r))
    if od:
        for meth in ["local", "fedavg", "fedprox"]:
            vals = [rr["mean_mae"] for rr in od["results"] if rr["method"] == meth]
            if vals:
                per_seed[meth] = vals
    nd = load(new_file(s, m, r))
    if nd:
        vals = [rr["mean_mae"] for rr in nd["results"] if rr["method"] == "fed_ca"]
        if vals:
            per_seed["fed_ca_fix"] = vals
        # Prefer newer local from CA-fix run if present
        vals_local = [rr["mean_mae"] for rr in nd["results"] if rr["method"] == "local"]
        if vals_local:
            per_seed["local"] = vals_local
    return per_seed


def load_tau_sweep():
    """Load tau_sweep_*.json files. Returns list sorted by tau.
    Each entry: {tau, scenario, mnar_method, missing_rate, mae_seeds, mae_mean, mae_std}.
    Note: tau sweep was run with the OLD (buggy) CA — kept as bug-discovery evidence.
    """
    entries = []
    for f in sorted(glob.glob(str(BASE / "tau_sweep_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8-sig"))
        taus = [rr.get("ca_tau") for rr in d.get("results", []) if rr.get("ca_tau")]
        if not taus:
            continue
        maes = [rr["mean_mae"] for rr in d["results"] if rr["method"] == "fed_ca"]
        if not maes:
            continue
        mn = statistics.mean(maes)
        sd = statistics.stdev(maes) if len(maes) > 1 else 0.0
        entries.append({
            "tau": taus[0],
            "scenario": d["scenario"],
            "mnar_method": d["mnar_method"],
            "missing_rate": d["missing_rate"],
            "mae_seeds": maes,
            "mae_mean": mn,
            "mae_std": sd,
        })
    entries.sort(key=lambda x: x["tau"])
    return entries


def load_all(with_centralized=False):
    data = {}
    for key in SETTINGS:
        s, m, r = key
        nd = load(new_file(s, m, r))
        od = load(old_recon_file(s, m, r))
        ca = load(old_ca_file(s, m, r))
        all_old = (od["results"] if od else []) + (ca["results"] if ca else [])
        all_new = nd["results"] if nd else []

        row = {}
        for method in ["local","fedavg","fedprox"]:
            mn, sd = get_stats(all_old, method)
            if mn is not None:
                row[method] = (mn, sd)
        mn, sd = get_stats(all_new, "fed_ca")
        if mn is not None:
            row["fed_ca_fix"] = (mn, sd)
        # local from new (more seeds in most cases)
        mn_new, sd_new = get_stats(all_new, "local")
        if mn_new is not None:
            row["local"] = (mn_new, sd_new)

        if with_centralized:
            cd = load(ceiling_file(s, m, r))
            if cd:
                mn, sd = get_stats(cd["results"], "centralized")
                if mn is not None:
                    row["centralized"] = (mn, sd)
        data[key] = row
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Figure A — Line chart: S1 quantile MAE vs ρ
# ══════════════════════════════════════════════════════════════════════════════
def fig_s1_line(data, with_centralized=False, save=True):
    rhos = [0.3, 0.5, 0.7]
    methods_main = ["local","fedavg","fedprox","fed_ca_fix"]
    if with_centralized:
        methods_main = ["centralized"] + methods_main

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    fig.suptitle("Figure A：S1（互補性）quantile MNAR — MAE vs 缺失率 ρ\n"
                 "（誤差線 = std，5 seeds）", fontsize=12, fontweight="bold", y=1.02)

    for ax, metric_key, ylabel, title_suffix in zip(
        axes, ["mae","rmse"], ["MAE","RMSE"], ["MAE","RMSE"]
    ):
        for method in methods_main:
            ys, errs = [], []
            for r in rhos:
                key = ("S1","quantile",r)
                val = data[key].get(method)
                if val:
                    ys.append(val[0])
                    errs.append(val[1])
                else:
                    ys.append(np.nan)
                    errs.append(0)
            ls = "--" if method == "local" else "-"
            mk = "o" if method not in ("local","centralized") else ("s" if method=="centralized" else "^")
            ax.errorbar(rhos, ys, yerr=errs,
                        label=LABEL[method], color=C[method],
                        marker=mk, linewidth=2, markersize=6, linestyle=ls,
                        capsize=4)
        ax.set_xlabel("缺失率 ρ", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"S1 quantile MNAR — {title_suffix}", fontsize=11)
        ax.set_xticks(rhos)
        ax.legend(framealpha=0.9, loc="upper left")

    plt.tight_layout()
    out = FIG / "figA_s1_quantile_line.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure B — Line chart: S4 quantile MAE vs ρ
# ══════════════════════════════════════════════════════════════════════════════
def fig_s4_line(data, with_centralized=False, save=True):
    rhos = [0.3, 0.5, 0.7]
    methods_main = ["local","fedavg","fedprox","fed_ca_fix"]
    if with_centralized:
        methods_main = ["centralized"] + methods_main

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Figure B：S4（無互補性）quantile MNAR — MAE vs 缺失率 ρ\n"
                 "（誤差線 = std，5 seeds）", fontsize=12, fontweight="bold", y=1.02)

    for ax, ylabel in zip(axes, ["MAE","RMSE"]):
        for method in methods_main:
            ys, errs = [], []
            for r in rhos:
                val = data[("S4","quantile",r)].get(method)
                ys.append(val[0] if val else np.nan)
                errs.append(val[1] if val else 0)
            ls = "--" if method == "local" else "-"
            mk = "o" if method not in ("local","centralized") else ("s" if method=="centralized" else "^")
            ax.errorbar(rhos, ys, yerr=errs,
                        label=LABEL[method], color=C[method],
                        marker=mk, linewidth=2, markersize=6, linestyle=ls,
                        capsize=4)
        # shade "FL is harmful" region (where FL > local)
        local_ys = [data[("S4","quantile",r)].get("local",(None,None))[0] for r in rhos]
        ax.fill_between(rhos, [y if y else 0 for y in local_ys],
                        [1.8]*3, alpha=0.05, color="red", label="_nolegend_")
        ax.set_xlabel("缺失率 ρ", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"S4 quantile MNAR — {ylabel}", fontsize=11)
        ax.set_xticks(rhos)
        ax.legend(framealpha=0.9)

    plt.tight_layout()
    out = FIG / "figB_s4_quantile_line.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure C — Bar chart: logit ρ=0.3 (S1 & S4)
# ══════════════════════════════════════════════════════════════════════════════
def fig_logit_bar(data, with_centralized=False, save=True):
    methods = ["local","fedavg","fedprox","fed_ca_fix"]
    if with_centralized:
        methods = ["centralized"] + methods

    scenarios = [("S1","logit",0.3), ("S4","logit",0.3)]
    scen_labels = ["S1（互補）logit ρ=0.3", "S4（無互補）logit ρ=0.3"]

    x = np.arange(len(methods))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle("Figure C：logit MNAR ρ=0.3 — S1 vs S4 各方法 MAE 比較\n"
                 "（誤差線 = std，5 seeds）", fontsize=12, fontweight="bold", y=1.02)

    for ax, key, slabel in zip(axes, scenarios, scen_labels):
        vals  = [data[key].get(m, (np.nan, 0)) for m in methods]
        means = [v[0] for v in vals]
        stds  = [v[1] for v in vals]
        colors = [C[m] for m in methods]
        bars = ax.bar(x, means, color=colors, width=0.55,
                      yerr=stds, error_kw=dict(capsize=5, linewidth=1.5),
                      zorder=3)
        # value labels
        for bar, mn in zip(bars, means):
            if not np.isnan(mn):
                ax.text(bar.get_x() + bar.get_width()/2, mn + 0.01,
                        f"{mn:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m].replace(" ","\n") for m in methods], fontsize=9)
        ax.set_ylabel("MAE", fontsize=11)
        ax.set_title(slabel, fontsize=11)
        ax.set_ylim(0, max([m for m in means if not np.isnan(m)]) * 1.25)

    plt.tight_layout()
    out = FIG / "figC_logit_bar.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure D — FL gain over local (% improvement) — grouped bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_fl_gain(data, with_centralized=False, save=True):
    methods = ["fedavg","fedprox","fed_ca_fix"]
    if with_centralized:
        methods = ["centralized"] + methods

    xlabels = ["S1-q\nρ=0.3","S1-q\nρ=0.5","S1-q\nρ=0.7","S1-logit\nρ=0.3",
               "S4-q\nρ=0.3","S4-q\nρ=0.5","S4-q\nρ=0.7","S4-logit\nρ=0.3"]

    n_groups = len(SETTINGS)
    n_methods = len(methods)
    x = np.arange(n_groups)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Figure D：各方法相對 Local-Only 的 MAE 改善率（%）\n"
                 "正值 = 比 Local 好，負值 = 比 Local 差", fontsize=12, fontweight="bold")

    for i, method in enumerate(methods):
        gains = []
        for key in SETTINGS:
            local_v = data[key].get("local")
            method_v = data[key].get(method)
            if local_v and method_v and not np.isnan(local_v[0]) and local_v[0] > 0:
                gain = (local_v[0] - method_v[0]) / local_v[0] * 100
            else:
                gain = np.nan
            gains.append(gain)

        offset = (i - n_methods/2 + 0.5) * width
        colors = [C[method] if not np.isnan(g) else "#cccccc" for g in gains]
        for j, (g, col) in enumerate(zip(gains, colors)):
            if not np.isnan(g):
                bar = ax.bar(x[j] + offset, g, width*0.9, color=col, zorder=3,
                             label=LABEL[method] if j == 0 else "_nolegend_")
                if abs(g) > 1.5:
                    va = "bottom" if g > 0 else "top"
                    ax.text(x[j] + offset, g + (0.5 if g > 0 else -0.5),
                            f"{g:+.1f}%", ha="center", va=va, fontsize=7.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1.2)
    # S1 / S4 divider
    ax.axvline(3.5, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(1.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 5,
            "← S1（互補性）", ha="center", fontsize=10, color="gray")
    ax.text(5.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 5,
            "S4（無互補）→", ha="center", fontsize=10, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("MAE 改善率 (%) vs Local", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    out = FIG / "figD_fl_gain_overview.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure E — CA fix 前後對比（舊 fed_ca vs 新 fed_ca_fix）
# ══════════════════════════════════════════════════════════════════════════════
def fig_ca_bugfix(data, save=True):
    xlabels = ["S1-q\nρ=0.3","S1-q\nρ=0.5","S1-q\nρ=0.7","S1-logit\nρ=0.3",
               "S4-q\nρ=0.3","S4-q\nρ=0.5","S4-q\nρ=0.7","S4-logit\nρ=0.3"]
    old_ca_data = {}
    for key in SETTINGS:
        s, m, r = key
        d = load(old_ca_file(s, m, r))
        if d:
            mn, sd = get_stats(d["results"], "fed_ca")
            old_ca_data[key] = (mn, sd)

    x = np.arange(len(SETTINGS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.suptitle("Figure E：CA 修正前後對比 — fed_ca（舊）vs fed_ca_fix（新）\n"
                 "（誤差線 = std，5 seeds，數值為 MAE）",
                 fontsize=12, fontweight="bold")

    old_means = [old_ca_data.get(k,(np.nan,0))[0] for k in SETTINGS]
    old_stds  = [old_ca_data.get(k,(np.nan,0))[1] for k in SETTINGS]
    new_means = [data[k].get("fed_ca_fix",(np.nan,0))[0] for k in SETTINGS]
    new_stds  = [data[k].get("fed_ca_fix",(np.nan,0))[1] for k in SETTINGS]

    b1 = ax.bar(x - width/2, old_means, width, label="fed_ca（舊版，有 bug）",
                color=C["fed_ca"], yerr=old_stds, error_kw=dict(capsize=4), zorder=3)
    b2 = ax.bar(x + width/2, new_means, width, label="fed_ca_fix（新版，修正後）",
                color=C["fed_ca_fix"], yerr=new_stds, error_kw=dict(capsize=4), zorder=3)

    for bars, means in [(b1, old_means),(b2, new_means)]:
        for bar, mn in zip(bars, means):
            if not np.isnan(mn):
                ax.text(bar.get_x()+bar.get_width()/2, mn+0.01,
                        f"{mn:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.axvline(3.5, color="gray", linestyle="--", alpha=0.6)
    ax.text(1.5, ax.get_ylim()[1]*0.97 if ax.get_ylim()[1]>0 else 0.1,
            "S1（互補）", ha="center", fontsize=10, color="gray")
    ax.text(5.5, ax.get_ylim()[1]*0.97 if ax.get_ylim()[1]>0 else 0.1,
            "S4（無互補）", ha="center", fontsize=10, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("MAE", fontsize=11)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    out = FIG / "figE_ca_bugfix_comparison.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure F — Fed-CA fix vs FedAvg delta (new version only)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ca_vs_fedavg(data, with_centralized=False, save=True):
    xlabels = ["S1-q\nρ=0.3","S1-q\nρ=0.5","S1-q\nρ=0.7","S1-logit\nρ=0.3",
               "S4-q\nρ=0.3","S4-q\nρ=0.5","S4-q\nρ=0.7","S4-logit\nρ=0.3"]

    deltas = []
    for key in SETTINGS:
        fa = data[key].get("fedavg")
        ca = data[key].get("fed_ca_fix")
        if fa and ca:
            deltas.append((fa[0] - ca[0]) / fa[0] * 100)  # positive = CA better
        else:
            deltas.append(np.nan)

    colors = [C["fed_ca_fix"] if d > 0 else C["fedavg"] for d in deltas]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Figure F：Fed-CA fix vs FedAvg — MAE 相對改善（%）\n"
                 "正值 = Fed-CA fix 優於 FedAvg，負值 = Fed-CA fix 較差",
                 fontsize=12, fontweight="bold")

    bars = ax.bar(range(len(SETTINGS)), deltas, color=colors, zorder=3)
    for bar, d in zip(bars, deltas):
        if not np.isnan(d):
            va = "bottom" if d >= 0 else "top"
            offset = 0.3 if d >= 0 else -0.3
            ax.text(bar.get_x()+bar.get_width()/2, d+offset,
                    f"{d:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(3.5, color="gray", linestyle="--", alpha=0.6)
    ax.text(1.5, max([d for d in deltas if not np.isnan(d)])*0.85,
            "S1（互補）", ha="center", fontsize=10, color="gray")
    ax.text(5.5, max([d for d in deltas if not np.isnan(d)])*0.85,
            "S4（無互補）", ha="center", fontsize=10, color="gray")

    patch_ca  = mpatches.Patch(color=C["fed_ca_fix"], label="Fed-CA fix 較好")
    patch_fa  = mpatches.Patch(color=C["fedavg"],     label="FedAvg 較好")
    ax.legend(handles=[patch_ca, patch_fa], framealpha=0.9)
    ax.set_xticks(range(len(SETTINGS)))
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Fed-CA fix 相對 FedAvg 改善率 (%)", fontsize=11)
    plt.tight_layout()
    out = FIG / "figF_ca_fix_vs_fedavg.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure G — Ceiling gap: grouped bar chart, all methods vs centralized
# ══════════════════════════════════════════════════════════════════════════════
def fig_ceiling_gap(data, save=True):
    xlabels = ["S1-q\nρ=0.3","S1-q\nρ=0.5","S1-q\nρ=0.7","S1-logit\nρ=0.3",
               "S4-q\nρ=0.3","S4-q\nρ=0.5","S4-q\nρ=0.7","S4-logit\nρ=0.3"]

    has_ceiling = any(data[k].get("centralized") for k in SETTINGS)
    if not has_ceiling:
        print("  [SKIP] figG: no centralized results yet.")
        return

    methods = ["local", "fedavg", "fed_ca_fix", "centralized"]
    n_methods = len(methods)
    x = np.arange(len(SETTINGS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15, 5.5))
    fig.suptitle("Figure G：各方法 MAE 全覽 vs 集中式天花板（Centralized Ceiling）\n"
                 "（誤差線 = std，5 seeds）", fontsize=12, fontweight="bold")

    for i, method in enumerate(methods):
        means = [data[k].get(method, (np.nan, 0))[0] for k in SETTINGS]
        stds  = [data[k].get(method, (np.nan, 0))[1] for k in SETTINGS]
        offset = (i - (n_methods - 1) / 2) * width
        hatch = "//" if method == "centralized" else None
        bars = ax.bar(x + offset, means, width,
                      color=C[method], label=LABEL[method],
                      yerr=stds, error_kw=dict(capsize=3, linewidth=1.2),
                      hatch=hatch, edgecolor="white" if hatch is None else C[method],
                      zorder=3, alpha=0.9)
        # value labels for centralized only (to avoid clutter)
        if method == "centralized":
            for bar, mn in zip(bars, means):
                if not np.isnan(mn):
                    ax.text(bar.get_x() + bar.get_width() / 2, mn + 0.015,
                            f"{mn:.3f}", ha="center", va="bottom",
                            fontsize=7, color=C["centralized"], fontweight="bold")

    # annotate the S1 ρ=0.7 "CA beats ceiling" point
    idx_s1q07 = 2  # index of S1 quantile 0.7 in SETTINGS
    ca_val = data[("S1","quantile",0.7)].get("fed_ca_fix")
    ceil_val = data[("S1","quantile",0.7)].get("centralized")
    if ca_val and ceil_val:
        ax.annotate("CA fix < Ceiling!\n(聯邦學習超越集中式)",
                    xy=(x[idx_s1q07] + (-0.5)*width, ca_val[0]),
                    xytext=(x[idx_s1q07] - 0.6, ca_val[0] - 0.12),
                    fontsize=8.5, color=C["fed_ca_fix"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["fed_ca_fix"], lw=1.5))

    ax.axvline(3.5, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(1.5, 0.02, "← S1（互補性）", ha="center", fontsize=10,
            color="gray", transform=ax.get_xaxis_transform())
    ax.text(5.5, 0.02, "S4（無互補）→", ha="center", fontsize=10,
            color="gray", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("MAE（↓ 越低越好）", fontsize=11)
    ax.legend(framealpha=0.9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    out = FIG / "figG_ceiling_gap.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure I — FedProx failure mode analysis (S4 quantile ρ=0.3/0.5/0.7)
#   Ported from archived plot_results.py::fig7_fedprox_failure.
#   Uses corrected CA-fix data for fed_ca_fix; fedprox per-seed values
#   come from mnar_recon_*.json (unchanged by CA bug).
# ══════════════════════════════════════════════════════════════════════════════
def fig_fedprox_failure(save=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Figure I：FedProx 失效模式分析 — S4 quantile MNAR\n"
                 "方形散點為各 seed，橫線為中位數",
                 fontsize=13, y=1.01)

    rhos = [0.3, 0.5, 0.7]
    titles = ["S4 quantile ρ=0.3\n（邊緣失效，不穩定）",
              "S4 quantile ρ=0.5\n（系統性崩壞）",
              "S4 quantile ρ=0.7\n（輕微崩壞）"]

    methods = ["local", "fedavg", "fedprox", "fed_ca_fix"]

    for ax, rho, title in zip(axes, rhos, titles):
        per_seed = load_per_seed("S4", "quantile", rho)
        all_vals, lbls, clrs = [], [], []
        for meth in methods:
            vals = per_seed.get(meth)
            if vals:
                all_vals.append(vals)
                lbls.append(LABEL[meth])
                clrs.append(C[meth])

        for xi, (vals, color) in enumerate(zip(all_vals, clrs)):
            jit = np.random.RandomState(0).uniform(-0.12, 0.12, len(vals))
            ax.scatter([xi + j for j in jit], vals, color=color, s=60,
                       zorder=5, alpha=0.85, edgecolors="white", linewidths=0.7)
            med = statistics.median(vals)
            ax.hlines(med, xi - 0.2, xi + 0.2, colors=color, linewidth=2.5, zorder=6)

        ax.set_xticks(np.arange(len(all_vals)))
        ax.set_xticklabels(lbls, fontsize=9)
        ax.set_ylabel("MAE")
        ax.set_title(title, fontsize=10)

        local_vals = per_seed.get("local")
        if local_vals:
            local_m = statistics.mean(local_vals)
            ax.axhline(local_m, color=C["local"], linestyle=":",
                       linewidth=1.5, alpha=0.7,
                       label=f"Local mean={local_m:.3f}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIG / "figI_fedprox_failure.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure J — Tau sweep (bug discovery evidence)
#   Ported from archived plot_results.py::fig10_tau_sweep.
#   Tau sweep was run with the OLD (buggy) CA. Result: MAE is flat across
#   tau values, clustered around FedAvg — this is the symptom that led to
#   discovering the fingerprint bug. Retained as methodology evidence.
# ══════════════════════════════════════════════════════════════════════════════
def fig_tau_sweep(data, tau_data, save=True):
    s1q05_tau = [e for e in tau_data
                 if e["scenario"] == "S1" and e["mnar_method"] == "quantile"
                 and e["missing_rate"] == 0.5]
    if not s1q05_tau:
        print("  [SKIP] figJ: no tau sweep data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Figure J：Tau Sweep 實驗（舊版 CA — Bug 發現過程記錄）\n"
        "[注意] 此實驗使用有 bug 的 CA（softmax + tau fingerprint）—"
        "結果不代表新版 CA 行為",
        fontsize=12, y=1.02,
    )

    taus  = [e["tau"]      for e in s1q05_tau]
    means = [e["mae_mean"] for e in s1q05_tau]
    stds  = [e["mae_std"]  for e in s1q05_tau]

    key = ("S1", "quantile", 0.5)
    fa = data[key].get("fedavg")
    fa_mean = fa[0] if fa else None
    ca_new  = data[key].get("fed_ca_fix")
    ca_new_mean = ca_new[0] if ca_new else None

    # Per-seed values for right panel
    per_seed = load_per_seed("S1", "quantile", 0.5)
    fa_vals = per_seed.get("fedavg", [])
    ca_new_vals = per_seed.get("fed_ca_fix", [])

    # ── Left: mean MAE vs tau ─────────────────────────────────────────────────
    ax = axes[0]
    ax.errorbar(taus, means, yerr=stds, marker="o", color="#e74c3c",
                linewidth=2, markersize=8, capsize=4,
                label="舊版 CA tau sweep (broken)")

    if fa_mean is not None:
        ax.axhline(fa_mean, color=C["fedavg"], linestyle="--", linewidth=2,
                   label=f"FedAvg ({fa_mean:.4f})")
    if ca_new_mean is not None:
        ax.axhline(ca_new_mean, color=C["fed_ca_fix"], linestyle="-", linewidth=2.5,
                   label=f"新版 CA (cafe fix) = {ca_new_mean:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("tau 值（log scale）")
    ax.set_ylabel("MAE (mean ± std, 3 seeds)")
    ax.set_title("舊版 CA：tau sweep 結果\n"
                 "（全部貼近 FedAvg → complementarity matrix 退化）",
                 fontsize=10.5)
    ax.legend(fontsize=9)

    ax.annotate(
        "tau 全程貼近 FedAvg：\ncomplementarity matrix 退化\n→ 發現 fingerprint bug",
        xy=(taus[len(taus)//2], statistics.mean(means)),
        xytext=(taus[0], statistics.mean(means) - 0.03),
        fontsize=8.5, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
    )

    # ── Right: FedAvg vs old CA vs new CA scatter ─────────────────────────────
    ax = axes[1]
    groups = {
        "FedAvg":              (fa_vals,            C["fedavg"]),
        "舊版 CA\n(broken)":   (means,              "#e74c3c"),
        "新版 CA\n(CAFE fix)": (ca_new_vals,        C["fed_ca_fix"]),
    }
    for xi, (label, (vals, color)) in enumerate(groups.items()):
        if not vals:
            continue
        jit = np.random.RandomState(xi).uniform(-0.12, 0.12, len(vals))
        ax.scatter([xi + j for j in jit], vals, color=color, s=65,
                   zorder=5, alpha=0.85, edgecolors="white", linewidths=0.7)
        ax.hlines(statistics.mean(vals), xi - 0.22, xi + 0.22,
                  colors=color, linewidth=2.5, zorder=6,
                  label=f"{label.replace(chr(10),' ')} mean={statistics.mean(vals):.4f}")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups.keys()), fontsize=9.5)
    ax.set_ylabel("MAE")
    ax.set_title("S1 q ρ=0.5：FedAvg vs 舊版 CA vs 新版 CA\n"
                 "（新版 CA 才真正低於 FedAvg）",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIG / "figJ_tau_sweep.png"
    if save:
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-centralized", action="store_true")
    args = parser.parse_args()

    wc = args.with_centralized
    data = load_all(with_centralized=wc)
    tau_data = load_tau_sweep()

    print("Generating figures...")
    fig_s1_line(data, with_centralized=wc)
    fig_s4_line(data, with_centralized=wc)
    fig_logit_bar(data, with_centralized=wc)
    fig_fl_gain(data, with_centralized=wc)
    fig_ca_bugfix(data)
    fig_ca_vs_fedavg(data)
    if wc:
        fig_ceiling_gap(data)
    fig_fedprox_failure()
    fig_tau_sweep(data, tau_data)

    print("\nAll figures saved to experiments/figures/")
    print("  figA — S1 quantile line chart (MAE & RMSE vs ρ)")
    print("  figB — S4 quantile line chart (MAE & RMSE vs ρ)")
    print("  figC — logit ρ=0.3 bar chart (S1 & S4)")
    print("  figD — FL gain % over local (all 8 settings)")
    print("  figE — CA fix 前後對比")
    print("  figF — Fed-CA fix vs FedAvg delta")
    if wc:
        print("  figG — Ceiling gap (with centralized)")
    print("  figI — FedProx failure mode analysis (S4 quantile)")
    print("  figJ — Tau sweep (bug discovery evidence)")


if __name__ == "__main__":
    main()
