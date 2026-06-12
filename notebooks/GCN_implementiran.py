#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model 2 — GCN (grafovska konvoluciona mreza)
============================================

Pisano paralelno sa `MLP_implementiran.py` i po ugovoru iz `GCN_PRAVILA.md`.

Sta je ISTO kao kod MLP-a (ne sme da se menja — vidi GCN_PRAVILA.md, tacka 2):
  * `loadData`  — isto ciscenje, jednom, pre podele  -> 8.882 molekula
  * `splitData` — 80/10/10, isto seme seed=42 -> iste molekule kao MLP
  * protokol obuke: Adam + MSE + rano zaustavljanje (patience=20)
  * metrike: R^2 i RMSE na testu

Sta je NOVO (samo ovo se menja — GCN_PRAVILA.md, tacka 3):
  * featurizacija: `smilesToGraph` umesto Morganovih otisaka
  * model: `SolubilityGCN` (3x GCNConv -> global_mean_pool -> head)
  * pakovanje u grupe: torch_geometric DataLoader (slaze vise grafova u jedan)
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

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")          # tisina za RDKit upozorenja

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# %% Putanja do podataka (prenosiva — radi i kod Vuka i kod Lazara)

def _find_project_root():
    """Pronalazi koren projekta (folder koji sadrzi data/raw).
    Radi i kao skripta i u Spyder celijama, na Windows-u i Linux-u."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())   # kada se pokrece kao fajl
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())            # kada se pokrece po celijama
    for start in candidates:
        for p in [start, *start.parents]:
            if (p / "data" / "raw").is_dir():
                return p
    raise RuntimeError("Ne mogu da pronadjem koren projekta MolGraph (nedostaje data/raw).")

PROJECT_ROOT = _find_project_root()
CSV_PATH     = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"
FIGURES_DIR  = PROJECT_ROOT / "figures"
MODELS_DIR   = PROJECT_ROOT / "models"


# %% 1. Ucitavanje i ciscenje  (ISTO kao MLP — ne menjati logiku)

def loadData(csvPath):
    """Ucitava AqSolDB CSV i zadrzava SAMO jedan povezan molekul po redu.

    Tacka '.' u SMILES-u razdvaja nepovezane delove (soli, kontrajoni, smese).
    Oni bi postali NEPOVEZAN graf, sto kvari prosledjivanje poruka u GCN-u, pa ih
    izbacujemo ovde — jednom, pre svake featurizacije. Time i sva tri modela rade
    na istom skupu molekula (pravicno poredjenje).
    Vraca (smiles, solubility) u istom redosledu."""
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


# %% 2. Featurizacija: SMILES -> graf  (NOVO)

# Atomski broj ide kao ONE-HOT (GCN_PRAVILA.md, tacka 5): element je KATEGORIJA,
# ne velicina. Sirov atomski broj (C=6, Cl=17) nadjaca male osobine i mreza
# poduci (R^2 ~ 0.53). One-hot to resava.
PERMITTED_ATOMS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se"]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def one_hot(value, choices):
    """One-hot vektor; poslednja pozicija je 'ostalo' (za vrednosti van liste)."""
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def atom_features(atom):
    """Osobine jednog atoma (cvor grafa), prema specifikaciji.
    atomski broj (one-hot) + valenca + aromaticnost + broj H + naelektrisanje."""
    feats = one_hot(atom.GetSymbol(), PERMITTED_ATOMS)   # one-hot elementa
    feats.append(float(atom.GetTotalValence()))          # valenca
    feats.append(float(atom.GetIsAromatic()))            # aromaticnost (0/1)
    feats.append(float(atom.GetTotalNumHs()))            # broj vodonika
    feats.append(float(atom.GetFormalCharge()))          # naelektrisanje
    return feats


def bond_features(bond):
    """Osobine jedne veze (grane grafa): tip veze (one-hot) + aromaticnost.
    GCN ovo NE koristi (usrednjava susede ravnopravno), ali ga gradimo i cuvamo
    jer ga GAT (Model 3) koristi — graf pisemo jednom, kompletno."""
    feats = one_hot(bond.GetBondType(), BOND_TYPES)      # tip veze (one-hot)
    feats.append(float(bond.GetIsAromatic()))            # aromaticnost (0/1)
    return feats


NUM_BOND_FEATURES = len(BOND_TYPES) + 1 + 1               # one-hot tip + 'ostalo' + aromat.


def smilesToGraph(smiles, y):
    """Pretvara jedan SMILES (+ njegov log S) u PyG `Data` objekat.

    Vraca graf sa:
      x          — matrica osobina atoma   [broj_atoma, broj_osobina]
      edge_index — spisak veza u OBA smera  [2, 2*broj_veza]
      edge_attr  — osobine veza             [2*broj_veza, NUM_BOND_FEATURES]
      y          — ciljna vrednost (log S) za ceo molekul
    Vraca None ako se SMILES ne moze parsirati (sigurnosna mreza)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src += [i, j]            # graf je neusmeren -> svaka veza u oba smera
        dst += [j, i]
        eattr += [bf, bf]

    if len(src) == 0:            # molekul bez veza (npr. jedan atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))
    data.smiles = smiles         # cuvamo SMILES (zatreba GAT-u za interpretaciju)
    return data


def buildGraphDataset(smiles, solubility):
    """Pravi listu grafova (PyG Data), red po red, preskacuci neparsirane molekule.
    Posle `loadData` ciscenja neparsiranih vise nema — ovo je sigurnosna mreza."""
    graphs = []
    for smi, sol in zip(smiles, solubility):
        g = smilesToGraph(smi, sol)
        if g is not None:
            graphs.append(g)
    return graphs


# %% 3. Podela 80/10/10  (ISTO seme kao MLP -> iste molekule)

def splitData(graphs, seed=SEED):
    """Deli listu grafova na trening / validaciju / test (80/10/10).
    Iste dve `train_test_split` operacije sa seed=42 kao kod MLP-a, pa su
    iste molekule u istim podskupovima -> poredjenje je posteno."""
    train, tmp = train_test_split(graphs, test_size=0.2, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed)
    return train, val, test


# %% 4. Model: SolubilityGCN  (NOVO)

class SolubilityGCN(nn.Module):
    """3x GCNConv (svaki atom usrednjava osobine suseda) -> global_mean_pool
    (atomi -> jedan vektor po molekulu) -> mali MLP head -> 1 broj (log S)."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)          # [broj_grafova, hidden]
        return self.head(x)                      # [broj_grafova, 1]


# %% 5. Jedan prolaz kroz trening  (ISTI protokol kao MLP, citanje grupe prilagodjeno grafovima)

def trainOneEpoch(model, loader, optimizer, lossFn):
    """Jedan prolaz kroz trening podatke, azurira tezine.
    Vraca prosecan trening gubitak za tu epohu."""
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


# %% 6. Procena bez ucenja  (validacija i test)

def evaluate(model, loader, lossFn):
    """Racuna gubitak na skupu BEZ azuriranja tezina. Koristi se za validaciju."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.x, batch.edge_index, batch.batch)
            total += lossFn(pred, batch.y.view(-1, 1)).item() * batch.num_graphs
    return total / len(loader.dataset)


# %% 7. fit — dirigent obuke (ISTI hiperparametri kao MLP)

def fit(model, train_set, val_set, epochs=500, patience=20, lr=1e-3, batch_size=64):
    """Trenira model. Ponavlja epohe, prati validacioni gubitak i rano zaustavlja
    (vracajuci najbolje tezine). Vraca istoriju {'train': [...], 'val': [...]}."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_weights = None
    waited = 0

    for epoch in range(1, epochs + 1):
        train_loss = trainOneEpoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:                 # poboljsanje -> sacuvaj ovu verziju
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


# %% 8. score — finalna ocena na testu (R^2 i RMSE)

def score(model, graphs):
    """Racuna R^2 i RMSE na datom skupu grafova. Koristi se jednom, na testu."""
    model.eval()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)   # shuffle=False -> red ostaje
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch.x, batch.edge_index, batch.batch).numpy().ravel())
            trues.append(batch.y.numpy().ravel())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return r2, rmse, y_true, y_pred


# %% 9. Grafici

def plot_history(history):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="trening")
    plt.plot(history["val"], label="validacija")
    plt.xlabel("epoha"); plt.ylabel("MSE gubitak"); plt.title("GCN — kriva obuke")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_gcn_kriva_obuke.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(y_true, y_pred, r2, rmse):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.4, edgecolors="none")
    lo, hi = y_true.min(), y_true.max()
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("izmereni log S"); plt.ylabel("predvidjeni log S")
    plt.title(f"GCN — test  (R²={r2:.3f}, RMSE={rmse:.2f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_gcn_rezultati.png", dpi=150, bbox_inches="tight")
    plt.show()


# %% 10. Ceo program (svaka linija je jedan poziv)

if __name__ == "__main__":
    smiles, solubility = loadData(CSV_PATH)              # 1. ucitaj + ocisti -> 8.882
    graphs = buildGraphDataset(smiles, solubility)       # 2. SMILES -> grafovi
    print("dataset:", len(graphs), "grafova |", graphs[0].x.shape[1], "osobina po atomu")

    train_set, val_set, test_set = splitData(graphs)     # 3. podela 80/10/10 (seed=42)
    print("train / val / test:", len(train_set), len(val_set), len(test_set))

    model = SolubilityGCN(in_dim=graphs[0].x.shape[1])   # 4. model
    history = fit(model, train_set, val_set)             # 5.-7. obuka

    r2, rmse, y_true, y_pred = score(model, test_set)    # 8. finalna ocena
    print(f"\nTEST  R² = {r2:.3f}   RMSE = {rmse:.3f} log mol/L")

    plot_history(history)                                # 9. grafici
    plot_predictions(y_true, y_pred, r2, rmse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)        # sacuvaj tezine
    torch.save(model.state_dict(), MODELS_DIR / "gcn_model.pt")
    print(f"Tezine sacuvane u: {MODELS_DIR / 'gcn_model.pt'}")
