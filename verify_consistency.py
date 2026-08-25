"""Cross-artifact consistency verifier: paper.tex/PAPER.md/README.md vs JSON ground truth."""

import json
import re
import sys

import generate_paper as gp


def main() -> int:
    data = gp.load_inputs()
    stats = gp.derived_stats(data)
    ch, ml = stats["ch"], stats["ml"]
    speedup = round(ml["id"]["latency_us"] / ch["id"]["latency_us"])
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    tex_shipped = open("paper.tex").read()
    md_shipped = open("PAPER.md").read()
    readme = open("README.md").read()

    for path, expected in (
        ("paper.tex", gp.build_paper_tex(data, stats)),
        ("PAPER.md", gp.build_paper_md(data, stats)),
    ):
        check(
            f"{path} byte-identical to JSON regeneration",
            open(path).read() == expected,
            f"{len(open(path).read())} bytes on disk",
        )

    m = re.search(r"Mean decision latency \.+ (\d+\.\d+) µs/query", readme)
    check(
        "README highlight latency == JSON",
        bool(m) and abs(float(m.group(1)) - round(ch["id"]["latency_us"], 3)) < 1e-9,
        m.group(1) if m else "missing",
    )
    m = re.search(r"Cost reduction vs always-frontier \.+ (\d+\.\d)%", readme)
    check(
        "README highlight ΔC == JSON",
        bool(m) and abs(float(m.group(1)) - round(ch["id"]["cost_reduction_pct"], 1)) < 1e-9,
        m.group(1) if m else "missing",
    )
    m = re.search(r"Speedup vs decision-tree router \.+ (\d+)×", readme)
    check(
        "README highlight speedup == JSON",
        bool(m) and int(m.group(1)) == speedup,
        m.group(1) if m else "missing",
    )

    for r in data["bench"]["results"]:
        pat = (
            f"| {r['id']['quality']:.4f} | {r['id']['cost_reduction_pct']:.2f}"
            f" | {r['ood']['quality']:.4f} |"
        )
        pat_bold = (
            f"| **{r['id']['quality']:.4f}** | **{r['id']['cost_reduction_pct']:.2f}**"
            f" | **{r['ood']['quality']:.4f}** |"
        )
        check(
            f"README Table 1 row values: {r['method']}",
            pat in readme or pat_bold in readme,
            pat,
        )

    for key, aggrec in data["abl"]["aggregates"].items():
        v = f"{aggrec['peak_fitness']['mean']:.2f}"
        s = f"{aggrec['peak_fitness']['std']:.2f}"
        check(
            f"Table 2 peak-fitness mean±std {v} ± {s} in PAPER.md ({key})",
            v in md_shipped and s in md_shipped and "±" in md_shipped,
        )
        g95 = f"{aggrec['gen_to_95pct_final']['mean']:.1f}"
        check(
            f"Table 2 gen→95% mean {g95} in PAPER.md ({key})",
            g95 in md_shipped,
        )

    for token in (
        f"{ch['id']['cost_reduction_pct']:.1f}",
        f"{ch['id']['latency_us']:.3f}",
        f"{ch['ood']['cost_reduction_pct']:.2f}",
        f"{stats['speedup_vs_ml']:.0f}",
    ):
        check(f"paper.tex headline token '{token}'", token in tex_shipped)
        check(f"PAPER.md headline token '{token}'", token in md_shipped)

    fails = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        line = ("PASS  " if ok else "FAIL  ") + name
        if not ok and detail:
            line += f"   [{detail}]"
        print(line)
    print(f"\n{len(checks) - len(fails)}/{len(checks)} consistency checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
