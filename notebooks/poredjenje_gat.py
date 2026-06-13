#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dodatni eksperiment — "dva GAT-a": obican vs GATv2 bez poveza
=============================================================

ODVOJENO od glavnog poredjenja tri modela. Pitanje za odbranu:

    Da li je obican GAT izgubio od GCN-a zato sto je paznja beskorisna na malim
    molekulima, ili zato sto smo mu paznju "zavezali ocima" (nije video veze)?

Trenira na ISTOJ podeli (seed=42) tri modela radi konteksta:
    GCN  (osnova, bez paznje)
    GAT  (paznja, ali NE vidi osobine veza)        <- nas originalni
    GATv2 (paznja koja VIDI osobine veza + GATv2)   <- "bez poveza"

Ako GATv2 stigne/prestigne GCN  -> paznja je bila zavezana ocima (dizajn).
Ako i dalje ~ kao GCN           -> niska povezanost malih molekula je pravi razlog.
"""

import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GCN_implementiran import (loadData, buildGraphDataset, SolubilityGCN,
                               fit as fit_plain, score as score_plain,
                               CSV_PATH, FIGURES_DIR)
from GAT_implementiran import SolubilityGAT
from GATv2_implementiran import SolubilityGATv2, fit as fit_edge, score as score_edge, NUM_BOND_FEATURES

SEED = 42


if __name__ == "__main__":
    smiles, solub = loadData(CSV_PATH)
    graphs = buildGraphDataset(smiles, solub)
    in_dim = graphs[0].x.shape[1]

    idx = np.arange(len(graphs))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=SEED)
    y_test = solub[idx_test]
    gtr = [graphs[i] for i in idx_tr]
    gval = [graphs[i] for i in idx_val]
    gtest = [graphs[i] for i in idx_test]
    print(f"train / val / test: {len(gtr)} / {len(gval)} / {len(gtest)}")

    print("\n=== GCN (osnova) ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    gcn = SolubilityGCN(in_dim=in_dim)
    fit_plain(gcn, gtr, gval)
    _, _, _, pred_gcn = score_plain(gcn, gtest)

    print("\n=== GAT (paznja, ne vidi veze) ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    gat = SolubilityGAT(in_dim=in_dim)
    fit_plain(gat, gtr, gval)                 # GAT forward je kompatibilan sa plain fit
    _, _, _, pred_gat = score_plain(gat, gtest)

    print("\n=== GATv2 (paznja VIDI veze) ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    gatv2 = SolubilityGATv2(in_dim=in_dim, edge_dim=NUM_BOND_FEATURES)
    fit_edge(gatv2, gtr, gval)
    _, _, _, pred_gv2 = score_edge(gatv2, gtest)

    preds = {"GCN": pred_gcn, "GAT": pred_gat, "GAT-v2\n(bez poveza)": pred_gv2}

    # klase rastvorljivosti (granice iz specifikacije)
    cls = np.digitize(y_test, [-4, -2, 0])
    labels = ["Nerastvorljivo\n(<-4)", "Slabo\n(-4..-2)",
              "Rastvorljivo\n(-2..0)", "Visoko\n(>0)"]
    counts = [int((cls == c).sum()) for c in range(4)]

    def rmse(p, t):
        return float(np.sqrt(np.mean((p - t) ** 2))) if len(t) else float("nan")

    print("\n================ UKUPNO (test) ================")
    print(f"{'model':<22} {'R2':>7} {'RMSE':>7}")
    for name, p in preds.items():
        print(f"{name.replace(chr(10),' '):<22} {r2_score(y_test, p):>7.3f} "
              f"{mean_squared_error(y_test, p)**0.5:>7.3f}")

    print("\n============ RMSE po klasi rastvorljivosti ============")
    print(f"{'klasa':<22} {'N':>5}   {'GCN':>6} {'GAT':>6} {'GATv2':>6}")
    for c in range(4):
        m = cls == c
        row = " ".join(f"{rmse(preds[name][m], y_test[m]):>6.3f}" for name in preds)
        print(f"{labels[c].replace(chr(10),' '):<22} {counts[c]:>5}   {row}")

    # grafik: ukupni R^2 i RMSE
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = [n.replace("\n", " ") for n in preds]
    r2s = [r2_score(y_test, preds[n]) for n in preds]
    rmses = [mean_squared_error(y_test, preds[n]) ** 0.5 for n in preds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#b2e061", "#fd7f6f", "#7eb0d5"]
    for ax, vals, ttl in zip(axes, [r2s, rmses], ["R²  (vise = bolje)", "RMSE  (manje = bolje)"]):
        bars = ax.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
        ax.set_title(ttl)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Dva GAT-a vs GCN  (ista podela seed=42)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "19_poredjenje_gatovi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafik sacuvan u: {FIGURES_DIR / '19_poredjenje_gatovi.png'}")
