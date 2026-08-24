"""Publication-ready figures from evolution_search.db: Pareto frontier + fitness trajectory."""

import sqlite3
import subprocess
import sys
from statistics import mean
from typing import Dict, List

DB_PATH = "evolution_search.db"
ISLAND_COLORS = {0: "tab:blue", 1: "tab:green", 2: "tab:orange"}


def ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not found -> installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "matplotlib"])
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])


def load_rows(query: str):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(query).fetchall()
    finally:
        con.close()


def non_dominated(rows) -> List[sqlite3.Row]:
    front = []
    for a in rows:
        dominated = any(
            b["code_hash"] != a["code_hash"]
            and b["quality"] >= a["quality"]
            and b["cost_reduction_pct"] >= a["cost_reduction_pct"]
            and (b["quality"] > a["quality"] or b["cost_reduction_pct"] > a["cost_reduction_pct"])
            for b in rows
        )
        if not dominated:
            front.append(a)
    return front


def figure_pareto(rows, out_path: str) -> None:
    import matplotlib.pyplot as plt

    passed = [r for r in rows if r["passed"] == 1]
    front = sorted(non_dominated(passed), key=lambda r: r["cost_reduction_pct"])
    champion = max(passed, key=lambda r: r["fitness"])

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sc = ax.scatter(
        [r["cost_reduction_pct"] for r in passed],
        [r["quality"] for r in passed],
        c=[r["generation"] for r in passed],
        cmap="viridis",
        s=26,
        alpha=0.55,
        edgecolors="none",
        label="evaluated candidates",
    )
    ax.step(
        [r["cost_reduction_pct"] for r in front],
        [r["quality"] for r in front],
        where="post",
        color="crimson",
        lw=1.4,
        alpha=0.85,
        label=f"Pareto frontier (n={len(front)})",
        zorder=4,
    )
    ax.scatter(
        [r["cost_reduction_pct"] for r in front],
        [r["quality"] for r in front],
        s=64,
        facecolors="none",
        edgecolors="crimson",
        lw=1.5,
        zorder=5,
    )
    ax.scatter(
        [champion["cost_reduction_pct"]],
        [champion["quality"]],
        marker="*",
        s=340,
        color="gold",
        edgecolors="black",
        lw=0.9,
        zorder=6,
        label=f"champion ({champion['code_hash'][:8]})",
    )
    ax.annotate(
        f"fit={champion['fitness']:.2f}",
        xy=(champion["cost_reduction_pct"], champion["quality"]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=9,
    )
    bar = fig.colorbar(sc, ax=ax, pad=0.02)
    bar.set_label("discovery generation")
    ax.set_xlabel(r"Cost Reduction $\Delta C$ (%)")
    ax.set_ylabel(r"Relative Quality $Q$")
    ax.set_title("Pareto Frontier of Evolved Routing Policies")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_trajectory(rows, out_path: str) -> None:
    import matplotlib.pyplot as plt

    per_island: Dict[int, Dict[int, List[float]]] = {}
    for r in rows:
        if r["passed"] != 1 or r["generation"] < 1:
            continue
        per_island.setdefault(r["island_id"], {}).setdefault(r["generation"], []).append(
            r["fitness"]
        )

    baseline_row = load_rows("SELECT fitness FROM candidates WHERE origin='baseline' LIMIT 1")
    baseline_fit = baseline_row[0]["fitness"] if baseline_row else None

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for isl in sorted(per_island):
        gens = sorted(per_island[isl])
        maxima = [max(per_island[isl][g]) for g in gens]
        means = [mean(per_island[isl][g]) for g in gens]
        color = ISLAND_COLORS.get(isl, f"C{isl}")
        ax.plot(gens, maxima, color=color, marker="o", ms=4, lw=1.8, label=f"island {isl} · max")
        ax.plot(
            gens,
            means,
            color=color,
            ls="--",
            marker="x",
            ms=5,
            lw=1.3,
            alpha=0.85,
            label=f"island {isl} · mean",
        )
    if baseline_fit is not None:
        ax.axhline(baseline_fit, color="grey", ls=":", lw=1.3)
        ax.text(
            10.15,
            baseline_fit,
            f"baseline\n{baseline_fit:.1f}",
            va="center",
            fontsize=8.5,
            color="grey",
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Search Progress per Island (MAP-Elites + Island Model)")
    ax.set_xticks(range(1, 11))
    ax.legend(ncol=3, fontsize=8.5, loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ensure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        }
    )
    rows = load_rows("SELECT * FROM candidates")
    if not rows:
        raise SystemExit("no candidate evaluations found; run run.py first")
    figure_pareto(rows, "pareto_frontier.png")
    print("saved pareto_frontier.png")
    figure_trajectory(rows, "fitness_trajectory.png")
    print("saved fitness_trajectory.png")


if __name__ == "__main__":
    main()
