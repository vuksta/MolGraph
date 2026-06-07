#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 03:18:03 2026

@author: vuk
"""

#%%

import io, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, AllChem, Draw, PandasTools
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from IPython.display import display


RDLogger.DisableLog("rdApp.*")
SEED = 42
np.random.seed(SEED)
CSV_PATH = "/home/vuk/Faks/MITNOP/MolGraph/data/raw/curated-solubility-dataset.csv"


#%%
df = pd.read_csv(CSV_PATH)
print("redovi x kolone: ", df.shape)
print("\nkolone: \n", ", ".join(df.columns))
print("\nnedostajuci SMILES / Rastvorljivost:", df["SMILES"].isna().sum(), "/", df["Solubility"].isna().sum())
print("duplikati SMILESa: ", df["SMILES"].duplicated().sum())
df.head(3)

#%%

print("log S (target) distribucija:")
print(df["Solubility"].describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
print("\n")


plt.figure(figsize=(7, 5))
sns.histplot(df["Solubility"], kde=True, color="royalblue", bins=30)

plt.title("Distribucija log S (Rastvorljivost)", fontsize=14, fontweight="bold")
plt.xlabel("log S", fontsize=12)
plt.ylabel("Frekvencija / Broj", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

#%%

def real_frags(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return sum(1 for f in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False) if f.GetNumHeavyAtoms() > 1)

multi = df["SMILES"].astype(str).str.contains(".", regex=False)
rc = df.loc[multi, "SMILES"].astype(str).apply(real_frags)
print(f"višekomponentni unosi: {multi.sum():,} ({100*multi.mean():.1f}%)")
print(f"  · soli (1 molekul + kontra-joni): {(rc == 1).sum():,}")
print(f"  · prave smeše (≥2 molekula)      : {(rc >= 2).sum():,}")
print(f"  · neorganske soli (bez pravog molekula): {(rc == 0).sum():,}")

keep = df["SMILES"].astype(str).apply(lambda s: ("." not in s) and (Chem.MolFromSmiles(s) is not None))
df = df[keep].reset_index(drop=True)
print(f"\nočišćen skup (jedan molekul): {len(df):,} redova")

#%%

THRESHOLD = math.log10(200 * 1e-6)
df["isSoluble"] = (df["Solubility"] > THRESHOLD).astype(int)
print(f"binarni prag (log S): {THRESHOLD:.2f}")
print(df["isSoluble"].value_counts().rename({1: "rastvorljiv", 0: "nerastvorljiv"}).to_string())

def sol_class(v):
    if v >= 0:  return "visoko rastvorljiv"
    if v >= -2: return "rastvorljiv"
    if v >= -4: return "slabo rastvorljiv"
    return "nerastvorljiv"

ORDER  = ["jako rastvorljiv", "rastvorljiv", "slabo rastvorljiv", "nerastvorljiv"]
COLORS = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]

df["class"] = pd.Categorical(df["Solubility"].apply(sol_class), categories=ORDER, ordered=True)
print("\nbroj po klasama (4 nivoa):")
print(df["class"].value_counts()[ORDER].to_string())

#%%

def pick(ascending, lo=8, hi=20):
    d = df.copy()
    d["heavy"] = d["SMILES"].apply(lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms())
    return d[(d.heavy >= lo) & (d.heavy <= hi)].sort_values("Solubility", ascending=ascending).iloc[0]

sol_ex, ins_ex = pick(False), pick(True)
mols = [Chem.MolFromSmiles(sol_ex.SMILES), Chem.MolFromSmiles(ins_ex.SMILES)]
Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(280, 220),
    legends=[f"Rastvorljiv  (log S={sol_ex.Solubility:.1f})", f"Nerastvorljiv  (log S={ins_ex.Solubility:.1f})"])


#%%

def atom_map(mol, weights, size=(360, 320)):
    d2d = rdMolDraw2D.MolDraw2DCairo(*size)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, list(weights), draw2d=d2d, colorMap="coolwarm")
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText()))

fig, axes = plt.subplots(2, 2, figsize=(8.5, 8))

for col, (tag, row) in enumerate([("rastvorljivo", sol_ex), ("nerastvorljivo", ins_ex)]):
    m = Chem.MolFromSmiles(row.SMILES)
    
    crippen = [c[0] for c in rdMolDescriptors._CalcCrippenContribs(m)]
    
    AllChem.ComputeGasteigerCharges(m)
    gast = [m.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(m.GetNumAtoms())]
    
    axes[0, col].imshow(atom_map(m, crippen))
    axes[0, col].set_title(f"{tag}\nlipofilnost po atomu (Crippen)", fontsize=10)
    
    axes[1, col].imshow(atom_map(m, gast))
    axes[1, col].set_title("parcijalno naelektrisanje po atomu (Gasteiger)", fontsize=10)
    
    for r in range(2): axes[r, col].axis("off")

plt.tight_layout(); plt.show()

#%%

def gallery(ascending, n=8, lo=4, hi=30):
    d = df.copy()
    d["heavy"] = d["SMILES"].apply(lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms())
    d = d[(d.heavy >= lo) & (d.heavy <= hi)].sort_values("Solubility", ascending=ascending).head(n)
    mols = [Chem.MolFromSmiles(s) for s in d.SMILES]
    return Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(190, 150),
                                legends=[f"logS={v:.1f}" for v in d.Solubility])

fig, axes = plt.subplots(2, 1, figsize=(11, 9))


axes[0].imshow(gallery(False))
axes[0].set_title("Najrastvorljiviji:  mali i polarni molekuli", fontsize=12)

axes[1].imshow(gallery(True))
axes[1].set_title("Najnerastvorljiviji: veliki, halogenovani ili aromatični molekuli", fontsize=12)

for a in axes: a.axis("off")
plt.tight_layout(); plt.show()

#%%

DESCR = [
    ("MolWt", "molekulska masa", 1000), 
    ("NumHDonors", "donori vodoničnih veza", 10),
    ("NumHAcceptors", "akceptori vodoničnih veza", 20), 
    ("HeavyAtomCount", "teški atomi", 100),
    ("NumValenceElectrons", "valentni elektroni", 400), 
    ("NumHeteroatoms", "heteroatomi", 30),
    ("NumAromaticRings", "aromatični prstenovi", 10)
]

sol, ins = df[df.isSoluble == 1], df[df.isSoluble == 0]
fig, axes = plt.subplots(2, 4, figsize=(15, 7)); axes = axes.ravel()

for ax, (col, label, clip) in zip(axes, DESCR):
    ax.hist(sol.loc[sol[col] < clip, col], bins=30, alpha=0.6, color="#2a9d8f", label="rastvorljivo")
    ax.hist(ins.loc[ins[col] < clip, col], bins=30, alpha=0.6, color="#e76f51", label="nerastvorljivo")
    ax.set_title(label, fontsize=11)

axes[0].legend(fontsize=9)
axes[-1].axis("off")

fig.suptitle("Kako se molekulski deskriptori razlikuju između rastvorljivih i nerastvorljivih jedinjenja", y=1.0)

plt.tight_layout(); plt.show()

#%%

DESC_COLS = ["MolWt","MolLogP","MolMR","HeavyAtomCount","NumHAcceptors","NumHDonors",
             "NumHeteroatoms","NumRotatableBonds","NumValenceElectrons","NumAromaticRings",
             "NumSaturatedRings","NumAliphaticRings","RingCount","TPSA","LabuteASA","BalabanJ","BertzCT"]

corr = df[DESC_COLS].corrwith(df["Solubility"], method="spearman").sort_values()

plt.figure(figsize=(7, 5))
plt.barh(range(len(corr)), corr.values, color=["#2a9d8f" if v > 0 else "#e76f51" for v in corr.values])
plt.yticks(range(len(corr)), corr.index, fontsize=8); plt.axvline(0, color="#333", lw=0.8)

plt.xlabel("Spirmanova korelacija sa log S")
plt.title("Koji deskriptori prate rastvorljivost?")

plt.tight_layout(); plt.show()

print(f"Najjači pojedinačni prediktor: MolLogP, korelacija {corr['MolLogP']:.2f}")

#%%

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))

a1.scatter(df["MolLogP"], df["Solubility"], s=5, alpha=0.25, color="#e76f51", edgecolors="none")
a1.set_xlabel("MolLogP (lipofilnost)")
a1.set_ylabel("log S")
a1.set_title(f"Masnije → manje rastvorljivo  (ρ={corr['MolLogP']:.2f})")

a2.scatter(df["TPSA"], df["Solubility"], s=5, alpha=0.25, color="#2a9d8f", edgecolors="none")
a2.set_xlim(0, 250)
a2.set_xlabel("TPSA (polarna površina molekula, Å²)")
a2.set_ylabel("log S")
a2.set_title(f"Veća polarna površina → više rastvorljivo  (ρ={corr['TPSA']:.2f})")

plt.tight_layout(); plt.show()

#%%

cm = df[DESC_COLS].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks(range(len(DESC_COLS))); ax.set_xticklabels(DESC_COLS, rotation=90, fontsize=7)
ax.set_yticks(range(len(DESC_COLS))); ax.set_yticklabels(DESC_COLS, fontsize=7)

fig.colorbar(im, fraction=0.046, pad=0.04).set_label("Spirmanova korelacija")

ax.set_title("Deskriptori nisu nezavisni (visoka multikolinearnost)")

plt.tight_layout(); plt.show()

#%%

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def fp(smi):
    m = Chem.MolFromSmiles(smi); a = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), a); return a

rng = np.random.RandomState(SEED)
sub = rng.choice(len(df), size=2000, replace=False)
X = np.stack([fp(s) for s in df["SMILES"].values[sub]])

emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(
        PCA(n_components=50, random_state=SEED).fit_transform(X))

plt.figure(figsize=(6.4, 5.4))
sc = plt.scatter(emb[:,0], emb[:,1], c=df["Solubility"].values[sub], cmap="viridis", s=10, alpha=0.75, edgecolors="none")
plt.xticks([]); plt.yticks([])

plt.title("Hemijski prostor (t-SNE otisaka prstiju) prema log S")

plt.colorbar(sc, fraction=0.046, pad=0.04).set_label("log S (rastvorljivost)")

plt.tight_layout(); plt.show()

#%%

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))

g = df["Group"].value_counts().sort_index()
a1.bar(g.index.astype(str), g.values, color="#4a6fa5")
a1.set_title("Nivo pouzdanosti (Grupa)")
a1.set_ylabel("broj molekula")

for i, v in enumerate(g.values): 
    a1.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)

a2.hist(df.loc[df["SD"] < 1, "SD"], bins=40, color="#e9c46a")
a2.set_title("Rasipanje između dupliranih merenja (SD < 1)")
a2.set_xlabel("standardna devijacija za log S")

plt.tight_layout(); plt.show()

print(f"Udeo sa niskim SD (≤0.1): {100*(df['SD'] <= 0.1).mean():.0f}%   |   Maksimalni SD: {df['SD'].max():.2f}")

#%%
