#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poredjenje tri modela: MLP vs GCN vs GAT
========================================

Trenira sva tri modela na ISTOJ podeli (seed=42, iste molekule) i racuna gresku
RAZLOZENU po klasama rastvorljivosti iz specifikacije:
    Visoko (>0) | Rastvorljivo (-2..0) | Slabo (-4..-2) | Nerastvorljivo (<-4)

Zasto: ukupni R^2 sva tri modela je blizak. Razlaganje po klasama pokazuje GDE se
modeli zaista razlikuju (cesto na nerastvorljivom "repu").

Grafovska featurizacija i modeli se uvoze iz GCN_/GAT_implementiran (DRY). MLP grana
(Morganovi otisci) je ovde data ukratko jer ogledava `MLP_implementiran.py`
(isti hiperparametri), a Vukov fajl se ne dira.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# da uvoz sestrinskih modula radi i iz Spyder-a i iz terminala
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GCN_implementiran import (loadData, buildGraphDataset, SolubilityGCN,
                               fit as graph_fit, score as graph_score,
                               CSV_PATH, FIGURES_DIR)
from GAT_implementiran import SolubilityGAT

SEED = 42


# %% MLP grana (ogledalo MLP_implementiran.py — isti hiperparametri)

N_BITS = 2048
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)


def fingerprint(smiles):
    """SMILES -> 2048-bitni Morganov otisak (radijus 2)."""
    v = np.zeros((N_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(_fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)), v)
    return v


class SolubilityMLP(nn.Module):
    def __init__(self, n_bits=N_BITS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_bits, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.layers(x)


def mlp_fit(model, Xtr, ytr, Xval, yval, epochs=500, patience=20, lr=1e-3, batch=64):
    """Isti protokol kao grafovski fit: Adam + MSE + rano zaustavljanje (patience=20)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr).view(-1, 1)),
        batch_size=batch, shuffle=True)
    Xv, yv = torch.tensor(Xval), torch.tensor(yval).view(-1, 1)
    best, bw, waited = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lf(model(Xv), yv).item()
        if vl < best:
            best, bw, waited = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
        if waited >= patience:
            break
    model.load_state_dict(bw)
    return model


def mlp_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy().ravel()


# %% Glavni tok

if __name__ == "__main__":
    # 1. Ucitaj jednom; napravi i otiske i grafove, poravnate po indeksu
    smiles, solub = loadData(CSV_PATH)
    graphs = buildGraphDataset(smiles, solub)
    fps = np.stack([fingerprint(s) for s in smiles])
    print(f"otisci: {fps.shape} | grafova: {len(graphs)}")

    # 2. Jedna podela na nivou indeksa -> sva tri modela vide ISTE molekule
    idx = np.arange(len(smiles))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=SEED)
    y_test = solub[idx_test]
    print(f"train / val / test: {len(idx_tr)} / {len(idx_val)} / {len(idx_test)}")

    gtr = [graphs[i] for i in idx_tr]
    gval = [graphs[i] for i in idx_val]
    gtest = [graphs[i] for i in idx_test]
    in_dim = graphs[0].x.shape[1]

    # 3. Treniraj sva tri (re-seed pre svakog da rezultati prate samostalne fajlove)
    print("\n=== MLP ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    mlp = SolubilityMLP()
    mlp_fit(mlp, fps[idx_tr], solub[idx_tr], fps[idx_val], solub[idx_val])
    pred_mlp = mlp_predict(mlp, fps[idx_test])

    print("\n=== GCN ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    gcn = SolubilityGCN(in_dim=in_dim)
    graph_fit(gcn, gtr, gval)
    _, _, _, pred_gcn = graph_score(gcn, gtest)

    print("\n=== GAT ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    gat = SolubilityGAT(in_dim=in_dim)
    graph_fit(gat, gtr, gval)
    _, _, _, pred_gat = graph_score(gat, gtest)

    preds = {"MLP": pred_mlp, "GCN": pred_gcn, "GAT": pred_gat}

    # 4. Klase rastvorljivosti (granice iz specifikacije)
    #    digitize: <-4 -> 0 | -4..-2 -> 1 | -2..0 -> 2 | >0 -> 3
    cls = np.digitize(y_test, [-4, -2, 0])
    labels = ["Nerastvorljivo\n(<-4)", "Slabo\n(-4..-2)",
              "Rastvorljivo\n(-2..0)", "Visoko\n(>0)"]
    counts = [int((cls == c).sum()) for c in range(4)]

    def rmse(p, t):
        return float(np.sqrt(np.mean((p - t) ** 2))) if len(t) else float("nan")

    # 5. Tabela: ukupne metrike + RMSE po klasi
    print("\n================ UKUPNO (test) ================")
    print(f"{'model':<6} {'R2':>7} {'RMSE':>7}")
    for name, p in preds.items():
        print(f"{name:<6} {r2_score(y_test, p):>7.3f} {mean_squared_error(y_test, p)**0.5:>7.3f}")

    print("\n============ RMSE po klasi rastvorljivosti ============")
    print(f"{'klasa':<22} {'N':>5} {'MLP':>7} {'GCN':>7} {'GAT':>7}")
    for c in range(4):
        m = cls == c
        row = "  ".join(f"{rmse(preds[name][m], y_test[m]):>5.3f}" for name in preds)
        print(f"{labels[c].replace(chr(10),' '):<22} {counts[c]:>5}  {row}")

    # 6. Grafik: grupisani stubici RMSE po klasi
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(4); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (name, p) in enumerate(preds.items()):
        rmses = [rmse(p[cls == c], y_test[cls == c]) for c in range(4)]
        bars = ax.bar(x + (k - 1) * w, rmses, w, label=name)
        for b, v in zip(bars, rmses):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (log mol/L)")
    ax.set_title("Greska po klasi rastvorljivosti  (test, ista podela seed=42)")
    ax.legend(title="model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_poredjenje_po_klasama.png", dpi=150, bbox_inches="tight")
    plt.show()

    # 7. Grafik: ukupni R^2 i RMSE
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(preds)
    r2s = [r2_score(y_test, preds[n]) for n in names]
    rmses_all = [mean_squared_error(y_test, preds[n]) ** 0.5 for n in names]
    for ax, vals, ttl in zip(axes, [r2s, rmses_all], ["R²  (vise = bolje)", "RMSE  (manje = bolje)"]):
        bars = ax.bar(names, vals, color=["#7eb0d5", "#b2e061", "#fd7f6f"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
        ax.set_title(ttl)
    fig.suptitle("Ukupno poredjenje (test skup)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_poredjenje_ukupno.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nGrafici sacuvani u: {FIGURES_DIR}")
