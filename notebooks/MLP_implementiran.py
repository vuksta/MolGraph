#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:58:42 2026

@author: vuk
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

SEED = 42
RADIUS = 2
N_BITS = 2048

np.random.seed(SEED)
torch.manual_seed(SEED)

def loadData (csvPath):
    df = pd.read_csv(csvPath)
    
    keep = df["SMILES"].astype(str).apply(lambda s:("." not in s) and (Chem.MolFromSmiles(s) is not None))
    
    print(f"Original : {len(df):,} molekula ")
    print(f"Odbaceno : {(~keep).sum():,} soli / smeše ({100*(~keep).mean():.1f}%)")
    df = df[keep].reset_index(drop=True)
    print(f"Ocisceni   : {len(df):,} molekula")
    
    smiles = df["SMILES"].astype(str).tolist()
    solubility = df["Solubility"].astype(float).to_numpy(dtype = np.float32)
    return smiles, solubility    

#smajli, rastvorljiv = loadData(r"/home/vuk/Faks/MITNOP/MolGraph/data/raw/curated-solubility-dataset.csv")

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, fpSize = N_BITS)

def smileToFingerprint(smiles):
    """ Pretvara jedan SMILE string u 2048 - bitni vektor otiska. Vraca NumPy array ili None ako molekul ne moze da se parsuje"""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    vector = np.zeros((N_BITS), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(_fp_gen.GetFingerprint(mol), vector)
    return vector
    
def buildDataSet(smiles, solubility):
    """
    Vraća X (N x 2048) i y (N,),održali usklađeni red po red, sa ispuštenim molekulima koji se ne mogu analizirati.
    """
    X, y = [], []
    for smi, sol in zip(smiles, solubility):
        fp = smileToFingerprint(smi)
        if fp is None:
            continue
        X.append(fp)
        y.append(sol)
    return np.stack(X), np.array(y, dtype=np.float32)

def splitData (X, y, seed = SEED):
    """Podela u Train, Test, Validate"""
    Xtrain, Xtmp, yTrain, yTmp = train_test_split(X,y, test_size = 0.2, random_state = SEED)
    Xval, Xtest, yval, ytest = train_test_split(Xtmp, yTmp, test_size= 0.5, random_state= SEED)
    return (Xtrain, yTrain), (Xval, yval), (Xtest, ytest)

class SolubilityMLP(nn.Module):
    def __init__(self, n_bits = N_BITS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_bits, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,1),
            )
        
    def forward(self,x):
        return self.layers(x)
        
def makeTensor(arr):
    """Helper funckija"""
    return torch.tensor(arr, dtype = torch.float32)

def trainOneEpoch(model, loader, optimizer, lossFn):
    """Pokreće jedan prolaz kroz train data, ažurirajući težine.
Vraća prosečan gubitak pri obučavanju za tu epohu."""
    model.train()
    total = 0.0
    for xBatch, yBatch in loader:
        optimizer.zero_grad()
        prediction = model(xBatch)
        loss = lossFn(prediction, yBatch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xBatch)
    return total/len(loader.dataset)        

def evaluate(model, X, y, lossFn):
    """Računa gubitak (loss) na skupu podataka BEZ ažuriranja težina.
    Koristi se za validaciju (nakon svake epohe) i za finalni test."""
    model.eval()
    with torch.no_grad():
        loss = lossFn(model(X), y)
    return loss.item()


def fit(model, train_set, val_set, epochs=500, patience=20, lr=1e-3, batch_size=64):
    """Train the model. Repeats epochs, watches validation loss, and stops early
    (restoring the best weights). Returns the {'train': [...], 'val': [...]} history."""
    X_train, y_train = train_set
    X_val, y_val = val_set

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # DataLoader serves the training data in shuffled batches of 64.
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(makeTensor(X_train), makeTensor(y_train).view(-1, 1)),
        batch_size=batch_size, shuffle=True)
    Xv, yv = makeTensor(X_val), makeTensor(y_val).view(-1, 1)

    history = {"train": [], "val": []}
    best_val = float("inf")     # best validation loss so far
    best_weights = None         # the weights that achieved it
    waited = 0                  # epochs since the last improvement

    for epoch in range(1, epochs + 1):
        train_loss = trainOneEpoch(model, loader, optimizer, loss_fn)
        val_loss = evaluate(model, Xv, yv, loss_fn)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:                 # improved -> save this version
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            waited = 0
        else:                                   # no improvement -> count toward stopping
            waited += 1

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train {train_loss:.3f} | val {val_loss:.3f}")
        if waited >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_weights)         
    return history
    
    
def score(model, X, y):
    model.eval()
    with torch.no_grad():
        prediction = model(makeTensor(X)).numpy().ravel()
    r2 = r2_score(y, prediction)
    rmse = mean_squared_error(y, prediction) ** 0.5
    return r2, rmse, prediction

def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Training curve")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_predictions(y_true, y_pred, r2, rmse):

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.4, edgecolors="none")
    lo, hi = y_true.min(), y_true.max()
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("measured log S"); plt.ylabel("predicted log S")
    plt.title(f"Test set  (R²={r2:.3f}, RMSE={rmse:.2f})")
    plt.tight_layout(); plt.show()
    
def _find_project_root():
    """Pronalazi koren projekta (folder koji sadrži data/raw).
    Radi i kao skripta i u Spyder ćelijama, na Windows-u i Linux-u."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())   # kada se pokreće kao fajl
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())            # kada se pokreće po ćelijama
    for start in candidates:
        for p in [start, *start.parents]:
            if (p / "data" / "raw").is_dir():
                return p
    raise RuntimeError("Ne mogu da pronađem koren projekta MolGraph (nedostaje data/raw).")

CSV_PATH = _find_project_root() / "data" / "raw" / "curated-solubility-dataset.csv"

smiles, solubility = loadData(CSV_PATH)          
X, y = buildDataSet(smiles, solubility)         
print("dataset:", X.shape, y.shape)

train_set, val_set, test_set = splitData(X, y)   # 3. split 80/10/10
print("train / val / test:", len(train_set[0]), len(val_set[0]), len(test_set[0]))

model = SolubilityMLP()                           
history = fit(model, train_set, val_set)           

X_test, y_test = test_set                          
r2, rmse, predictions = score(model, X_test, y_test)
print(f"\nTEST  R² = {r2:.3f}   RMSE = {rmse:.3f} log mol/L")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    