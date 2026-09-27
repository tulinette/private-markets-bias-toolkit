"""
Generates the two figures referenced in the README:
  1. Kaplan-Meier survival curve
  2. Bar chart of 1-year refile hazard by cohort

Run from the repo root, with the venv active and data already downloaded:
    python3 plot_results.py
"""
import os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from biastk.load_panel import load_all_quarters
from biastk.survival_analysis import build_duration_table, compare_cohorts_by_year

OUT_DIR = "figures"


def make_km_plot(kmf):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.axvline(3, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Years since first Form D filing")
    ax.set_ylabel("Estimated probability of no follow-on filing yet")
    ax.set_title("Kaplan-Meier estimate: time to follow-on filing")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/kaplan_meier_curve.png", dpi=150)
    print(f"saved {OUT_DIR}/kaplan_meier_curve.png")


def make_cohort_plot(cohort_probs):
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = list(cohort_probs.keys())
    values = list(cohort_probs.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#3b6ea5")
    ax.set_ylabel("1-year refile probability")
    ax.set_title("Early follow-on financing has declined by cohort")
    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cohort_hazard_comparison.png", dpi=150)
    print(f"saved {OUT_DIR}/cohort_hazard_comparison.png")


if __name__ == "__main__":
    panel = load_all_quarters()
    panel = panel[panel["is_operating_company"]]
    as_of = panel["FILING_DATE"].max()

    table = build_duration_table(panel, as_of)
    table = table[table["duration_years"] > 0]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=table["duration_years"], event_observed=table["event_observed"])
    make_km_plot(kmf)

    cohort_probs = compare_cohorts_by_year(table)
    make_cohort_plot(cohort_probs)
