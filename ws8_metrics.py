"""WS8 benchmark metrics for the Primagaz full application.

Usage:
    python ws8_metrics.py results/ws8/results_13_44.xlsx
"""
import sys
import numpy as np
import pandas as pd


def to_kg(col):
    # "10.7?" -> 10.7, "?" / empty -> NaN, "10,8" -> 10.8
    s = col.astype(str).str.replace(",", ".").str.replace("?", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def metrics(df):
    n = len(df)
    lector = to_kg(df["tarra"])
    ref = lector.fillna(to_kg(df["PR_tarra"]))  # lector tare, operator tare if lector missing
    ai_tare = to_kg(df["AI_tarra"])
    ai_push = df["AI_push_or_fill"].str.strip().str.lower() == "push"
    pr_push = df["PR_push_or_fill"].str.strip().str.lower() == "push"

    # 1. Dangerous overfill: AI fills to (AI tare + net), so the bottle gets
    #    (AI tare - real tare) too much gas. Dangerous if > 0.5 kg.
    overfill = ai_tare - lector
    dangerous = (~ai_push) & (overfill > 0.5)
    filled_without_tare = (~ai_push) & ai_tare.isna()

    # 2. Push F3 (push = positive class)
    tp = (ai_push & pr_push).sum()
    recall = tp / pr_push.sum() if pr_push.sum() else np.nan
    precision = tp / ai_push.sum() if ai_push.sum() else np.nan
    b2 = 3 ** 2
    f3 = (1 + b2) * precision * recall / (b2 * precision + recall) if tp else 0.0

    # 3. RMSE tare in grams, only on bottles where AI read a tare
    read = ai_tare.notna() & ref.notna()
    err_g = (ai_tare[read] - ref[read]) * 1000
    rmse = np.sqrt((err_g ** 2).mean()) if read.any() else np.nan

    print(f"Bottles presented:            {n}")
    print(f"1. Dangerous overfills:       {dangerous.sum()}/{n} = {100 * dangerous.mean():.1f} %")
    if filled_without_tare.any():
        print(f"   (+ {filled_without_tare.sum()} filled with no AI tare - check your decision rule)")
    print(f"2. Pushed by operator:        {pr_push.sum()}")
    print(f"   Pushed by AI:              {ai_push.sum()}  (correct: {tp})")
    print(f"   Recall push / precision:   {100 * recall:.1f} % / {100 * precision:.1f} %")
    print(f"   F3:                        {f3:.3f}")
    print(f"3. RMSE tare:                 {rmse:.0f} g on {read.sum()} bottles with an AI tare "
          f"(mean error {err_g.mean():+.0f} g)")


if __name__ == "__main__":
    path = sys.argv[1]
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    metrics(df)