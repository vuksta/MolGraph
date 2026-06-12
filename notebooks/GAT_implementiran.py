#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model 3 — GAT (grafovska mreza sa mehanizmom paznje)
====================================================

Pisano paralelno sa `GCN_implementiran.py`, po istom ugovoru (GCN_PRAVILA.md).
GAT je "GCN sa GATConv umesto GCNConv + glave paznje": ista featurizacija, ista
podela (seed=42), isti protokol obuke. Menja se SAMO model.

Dodatak koji GCN nema — INTERPRETABILNOST:
  * model vraca i tezine paznje (`return_attention_weights=True`),
  * `visualize_attention` ih crta nazad na molekul (koje veze model smatra vaznim).
To je deo specifikacije: hidrofilne grupe treba da nose vece tezine.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# %% Putanja do podataka (prenosiva)

def _find_project_root():
    """Pronalazi koren projekta (folder koji sadrzi data/raw)."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())
    for start in candidates:
        for p in [start, *start.parents]:
            if (p / "data" / "raw").is_dir():
                return p
    raise RuntimeError("Ne mogu da pronadjem koren projekta MolGraph (nedostaje data/raw).")

PROJECT_ROOT = _find_project_root()
CSV_PATH     = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"
FIGURES_DIR  = PROJECT_ROOT / "figures"
MODELS_DIR   = PROJECT_ROOT / "models"


# %% 1. Ucitavanje i ciscenje  (ISTO kao MLP/GCN)

def loadData(csvPath):
    """Ucitava AqSolDB i zadrzava samo jedan povezan, parsiv molekul po redu
    (izbacuje SMILES sa '.'). Rezultat: 8.882 molekula. Vraca (smiles, solubility)."""
    import pandas as pd
    df = pd.read_csv(csvPath)
    keep = df["SMILES"].astype(str).apply(
        lambda s: ("." not in s) and (Chem.MolFromSmiles(s) is not None))
    print(f"Sirovo   : {len(df):,} molekula")
    print(f"Odbaceno : {(~keep).sum():,} soli / smese ({100*(~keep).mean():.1f}%)")
    df = df[keep].reset_index(drop=True)
    print(f"Ocisceno : {len(df):,} molekula")
    smiles = df["SMILES"].astype(str).tolist()
    solubility = df["Solubility"].astype(float).to_numpy(dtype=np.float32)
    return smiles, solubility


# %% 2. Featurizacija: SMILES -> graf  (ISTO kao GCN — atomski broj kao one-hot)

PERMITTED_ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se"]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
NUM_BOND_FEATURES = len(BOND_TYPES) + 1 + 1


def one_hot(value, choices):
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def atom_features(atom):
    """atomski broj (one-hot) + valenca + aromaticnost + broj H + naelektrisanje."""
    feats = one_hot(atom.GetSymbol(), PERMITTED_ATOMS)
    feats.append(float(atom.GetTotalValence()))
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.GetTotalNumHs()))
    feats.append(float(atom.GetFormalCharge()))
    return feats


def bond_features(bond):
    """tip veze (one-hot) + aromaticnost. GAT OVO koristi (za razliku od GCN-a)."""
    feats = one_hot(bond.GetBondType(), BOND_TYPES)
    feats.append(float(bond.GetIsAromatic()))
    return feats


def smilesToGraph(smiles, y):
    """SMILES -> PyG Data (x, edge_index u oba smera, edge_attr, y). None ako ne parsira."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]; dst += [j, i]; eattr += [bf, bf]
    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))
    data.smiles = smiles
    return data


def buildGraphDataset(smiles, solubility):
    graphs = []
    for smi, sol in zip(smiles, solubility):
        g = smilesToGraph(smi, sol)
        if g is not None:
            graphs.append(g)
    return graphs


# %% 3. Podela 80/10/10  (ISTO seme kao MLP/GCN)

def splitData(graphs, seed=SEED):
    train, tmp = train_test_split(graphs, test_size=0.2, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed)
    return train, val, test


# %% 4. Model: SolubilityGAT  (NOVO — vise glava paznje + ispis tezina paznje)

class SolubilityGAT(nn.Module):
    """3x GATConv (svaki atom NE usrednjava susede ravnopravno, vec uci koliko
    koji sused doprinosi) -> global_mean_pool -> head.

    `heads` paralelnih glava paznje daje stabilnost (kao u radu o GAT-u).
    Forward opciono vraca i tezine paznje prve konvolucije radi interpretacije."""

    def __init__(self, in_dim, hidden=32, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=0.2)
        # poslednja konvolucija: jedna glava (concat=False -> usrednjava glave)
        self.conv3 = GATConv(hidden * heads, hidden * heads, heads=1,
                             concat=False, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden * heads, hidden * heads // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden * heads // 2, 1),
        )

    def forward(self, x, edge_index, batch, return_attention=False):
        # prva konvolucija opciono vraca (edge_index_sa_petljama, alpha[E, heads])
        x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.head(x)
        return (out, att1) if return_attention else out


# %% 5.-8. Obuka i ocena  (ISTI protokol kao GCN; forward kompatibilan)

def trainOneEpoch(model, loader, optimizer, lossFn):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = lossFn(pred, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def evaluate(model, loader, lossFn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            total += lossFn(pred, batch.y.view(-1, 1)).item() * batch.num_graphs
    return total / len(loader.dataset)


def fit(model, train_set, val_set, epochs=500, patience=20, lr=1e-3, batch_size=64):
    """Isti protokol kao MLP/GCN: Adam + MSE + rano zaustavljanje (patience=20)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    history = {"train": [], "val": []}
    best_val, best_weights, waited = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_loss = trainOneEpoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        history["train"].append(train_loss); history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train {train_loss:.3f} | val {val_loss:.3f}")
        if waited >= patience:
            print(f"rano zaustavljanje na epohi {epoch}")
            break
    model.load_state_dict(best_weights)
    return history


def score(model, graphs):
    """R^2 i RMSE na testu."""
    model.eval()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch.x, batch.edge_index, batch.batch).numpy().ravel())
            trues.append(batch.y.numpy().ravel())
    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5, y_true, y_pred


# %% 9. Grafici

def plot_history(history):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="trening")
    plt.plot(history["val"], label="validacija")
    plt.xlabel("epoha"); plt.ylabel("MSE gubitak"); plt.title("GAT — kriva obuke")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_gat_kriva_obuke.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(y_true, y_pred, r2, rmse):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.4, edgecolors="none")
    lo, hi = y_true.min(), y_true.max()
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("izmereni log S"); plt.ylabel("predvidjeni log S")
    plt.title(f"GAT — test  (R²={r2:.3f}, RMSE={rmse:.2f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_gat_rezultati.png", dpi=150, bbox_inches="tight")
    plt.show()


# %% 10. Interpretabilnost: tezine paznje nacrtane na molekulu  (NOVO — srz GAT-a)

def visualize_attention(model, smiles, fname, title=None):
    """Crta molekul sa vezama obojenim po nauceno) tezini paznje.
    Zelena = visoka paznja. Ocekujemo da hidrofilne grupe (O, N) budu istaknute."""
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    g = smilesToGraph(smiles, 0.0)
    model.eval()
    with torch.no_grad():
        batch = torch.zeros(g.x.size(0), dtype=torch.long)
        _, (att_ei, alpha) = model(g.x, g.edge_index, batch, return_attention=True)
    alpha = alpha.mean(dim=1).numpy()           # prosek preko glava -> jedna vrednost po grani

    # nase grane su prvih 2*broj_veza (oba smera); ostalo su self-petlje (GATConv ih dodaje)
    nb = mol.GetNumBonds()
    bond_imp = np.array([(alpha[2 * b] + alpha[2 * b + 1]) / 2 for b in range(nb)])
    if bond_imp.max() > bond_imp.min():         # normalizacija na [0,1]
        norm = (bond_imp - bond_imp.min()) / (bond_imp.max() - bond_imp.min())
    else:
        norm = np.zeros_like(bond_imp)

    cmap = matplotlib.colormaps["Greens"]
    hl_bonds = list(range(nb))
    hl_colors = {b: tuple(cmap(0.25 + 0.75 * norm[b])[:3]) for b in range(nb)}

    d = rdMolDraw2D.MolDraw2DCairo(520, 420)
    if title:
        d.drawOptions().legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, legend=title or smiles,
        highlightAtoms=[], highlightBonds=hl_bonds, highlightBondColors=hl_colors)
    d.FinishDrawing()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIGURES_DIR / fname, "wb") as f:
        f.write(d.GetDrawingText())
    print(f"  -> paznja sacuvana: {fname}")


# %% 11. Ceo program

if __name__ == "__main__":
    smiles, solubility = loadData(CSV_PATH)
    graphs = buildGraphDataset(smiles, solubility)
    print("dataset:", len(graphs), "grafova |", graphs[0].x.shape[1], "osobina po atomu")

    train_set, val_set, test_set = splitData(graphs)
    print("train / val / test:", len(train_set), len(val_set), len(test_set))

    model = SolubilityGAT(in_dim=graphs[0].x.shape[1])
    history = fit(model, train_set, val_set)

    r2, rmse, y_true, y_pred = score(model, test_set)
    print(f"\nTEST  R² = {r2:.3f}   RMSE = {rmse:.3f} log mol/L")

    plot_history(history)
    plot_predictions(y_true, y_pred, r2, rmse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "gat_model.pt")
    print(f"Tezine sacuvane u: {MODELS_DIR / 'gat_model.pt'}")

    # Interpretabilnost: rastvorljiv polarni molekul vs nerastvorljiv nepolarni
    print("\nVizualizacija paznje:")
    visualize_attention(model, "OC[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
                        "10_gat_paznja_glukoza.png", title="glukoza (rastvorljiva)")
    visualize_attention(model, "CC(=O)Oc1ccccc1C(=O)O",
                        "11_gat_paznja_aspirin.png", title="aspirin")
    visualize_attention(model, "c1ccc2ccccc2c1",
                        "12_gat_paznja_naftalen.png", title="naftalen (nerastvorljiv)")
