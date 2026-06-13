
"""
Faza istrazivanja - Analiza podataka
=====================================================

"""
import warnings
warnings.filterwarnings("ignore")
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from pathlib import Path
 
print("Biblioteke uspešno uvezene.")
 
# %% Konfiguracija
 
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

PROJECT_ROOT = _find_project_root()
DATA_PATH   = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
SEED        = 42
SPLIT_VAL   = 0.10      # 10% test skup
SPLIT_TEST  = 0.1111    # 10% od preostalih 90% ≈ 10% ukupno
 
# Kolone sa unapred izračunatim 2D deskriptorima
DESCRIPTOR_COLS = [
    "MolWt", "MolLogP", "MolMR", "HeavyAtomCount",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRotatableBonds", "NumValenceElectrons",
    "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "RingCount", "TPSA", "LabuteASA", "BalabanJ", "BertzCT",
]
 
# Granice kategorija rastvorljivosti prema specifikaciji projekta
SOLUBILITY_BINS   = [-np.inf, -4, -2, 0, np.inf]
SOLUBILITY_LABELS = [
    "Nerastvorljivo (<-4)",
    "Slabo rastvorljivo (-4 do -2)",
    "Rastvorljivo (-2 do 0)",
    "Visoko rastvorljivo (>0)",
]
 
# Seaborn tema
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")
 
# Kreiranje direktorijuma za grafike (ako ne postoji)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
print(f"Konfiguracija postavljena. Grafici će biti sačuvani u: {FIGURES_DIR}")
 
# %% Pomoćne funkcije
 
def save(fig, name):
    """Čuva figuru na disk i zatvara je."""
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → sačuvano: {path.name}")
 
# %% [markdown]
# ## 1. Učitavanje i filtriranje
#
# Uklanjamo sve SMILES zapise koji sadrže `.` (tačku).
# Tačka u SMILES notaciji razdvaja nepovezane fragmente —
# što obuhvata soli, mešavine i metalne komplekse.
# Nijedan od ta tri tipa ne može se smisleno predstaviti kao jedan molekulski graf.
 
# %% 1. Učitavanje i filtriranje
 
raw = pd.read_csv(DATA_PATH)
print(f"Sirovi skup podataka   : {len(raw):,} redova × {raw.shape[1]} kolona")
 
# Maska za višekomponentne SMILES (sadrže tačku)
multi_mask = raw["SMILES"].str.contains(".", regex=False)
dropped    = raw[multi_mask]
df         = raw[~multi_mask].reset_index(drop=True)
 
# Klasifikacija uklonjenih molekula
pat_metals  = r"\[(?:Na|K|Ca|Mg|Zn|Fe|Cu|Al|Li)[+]"
pat_charged = r"\[.*?[+-]"
n_metals  = dropped["SMILES"].str.contains(pat_metals,  regex=True).sum()
n_charged = dropped["SMILES"].str.contains(pat_charged, regex=True).sum()
 
print(f"Uklonjeni (višekomponentni): {multi_mask.sum():,}  ({100*multi_mask.mean():.1f}%)")
print(f"  · metalni kompleksi / neorganske soli : {n_metals}")
print(f"  · bilo koji naelektrisani fragment    : {n_charged}")
print(f"Očišćen skup podataka  : {len(df):,} redova")
 
# %% [markdown]
# ## 2. Osnovna provera podataka
#
# Proveravamo nedostajuće vrednosti i duplikate.
# AqSolDB je već kuriran skup — očekujemo čiste podatke.
 
# %% 2. Osnovna provera podataka
 
missing = df.isna().sum()
print("Nedostajuće vrednosti :", "nema ✓" if missing.sum() == 0 else missing[missing > 0].to_string())
print(f"Duplirani SMILES       : {df['SMILES'].duplicated().sum()}")
print(f"Duplirani InChIKey     : {df['InChIKey'].duplicated().sum()}")
print(f"\nOblik skupa podataka   : {df.shape[0]:,} redova × {df.shape[1]} kolona")
print("\nTipovi podataka po koloni:")
print(df.dtypes.to_string())
 
# %% [markdown]
# ## 3. Raspodela ciljne promenljive
#
# Ciljna promenljiva je **log S** (logaritam rastvorljivosti u mol/L).
# Vrednosti od ~−13 do ~+2, srednja vrednost oko −3.
 
# %% 3. Raspodela ciljne promenljive — statistika
 
sol = df["Solubility"]
 
print("Osnovna statistika:")
print(sol.describe().round(3).to_string())
 
skew = stats.skew(sol)
kurt = stats.kurtosis(sol)
_, p_norm = stats.shapiro(sol.sample(min(len(sol), 5000), random_state=SEED))
 
print(f"\nAsimetrija (skewness)  : {skew:.3f}   (0 = savršeno simetrično)")
print(f"Spljoštenost (kurtosis): {kurt:.3f}   (0 = normalna raspodela)")
print(f"Shapiro–Wilk p-vrednost: {p_norm:.4f}  → {'nije normalna raspodela' if p_norm < 0.05 else 'normalna raspodela'}")
 
# %% 3. Raspodela ciljne promenljive — grafici
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Levo: histogram + KDE
ax = axes[0]
sns.histplot(sol, bins=60, kde=True, color=PALETTE[0], ax=ax, line_kws={"lw": 2})
ax.axvline(sol.mean(),   color="crimson", linestyle="--", lw=1.5, label=f"srednja vred. {sol.mean():.2f}")
ax.axvline(sol.median(), color="crimson", linestyle=":",  lw=1.5, label=f"medijana {sol.median():.2f}")
ax.set_xlabel("log S  (mol/L)")
ax.set_ylabel("Broj molekula")
ax.set_title("Raspodela rastvorljivosti")
ax.legend()
 
# Desno: kategorije rastvorljivosti
df["sol_category"] = pd.cut(sol, bins=SOLUBILITY_BINS, labels=SOLUBILITY_LABELS)
counts = df["sol_category"].value_counts().reindex(SOLUBILITY_LABELS)
 
ax2 = axes[1]
bars = ax2.bar(range(len(counts)), counts.values,
               color=sns.color_palette("RdYlGn", len(counts))[::-1], edgecolor="white")
ax2.set_xticks(range(len(counts)))
ax2.set_xticklabels(SOLUBILITY_LABELS, rotation=20, ha="right")
ax2.set_ylabel("Broj molekula")
ax2.set_title("Kategorije rastvorljivosti (prema specifikaciji)")
for bar, v in zip(bars, counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"{v}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", fontsize=9)
 
fig.tight_layout()
save(fig, "01_raspodela_ciljne.png")
 
print("\nBroj molekula po kategoriji:")
for label, count in counts.items():
    print(f"  {label:<35s}: {count:4d}  ({100*count/len(df):.1f}%)")
 
# %% [markdown]
# ## 4. Grupe pouzdanosti
#
# AqSolDB dodeljuje svako jedinjenje jednoj od 5 grupa prema broju izvora i slaganju merenja:
#
# | Grupa | Izvora | Slaganje | Napomena |
# |-------|--------|----------|----------|
# | G1    | 1      | —        | Nevalidovano (nema sa čim porediti) |
# | G2    | 2      | ✗        | SD ≥ 0.5, rezultati se razlikuju |
# | G3    | 2      | ✓        | SD < 0.5, rezultati se slažu |
# | G4    | 3+     | ✗        | SD ≥ 0.5, rezultati se razlikuju |
# | G5    | 3+     | ✓        | SD < 0.5 — **najpouzdaniji** |
 
# %% 4. Grupe pouzdanosti — statistika
 
group_stats = df.groupby("Group").agg(
    broj=("Solubility", "size"),
    sd_medijana=("SD", "median"),
    sd_max=("SD", "max"),
    occ_max=("Ocurrences", "max"),
    sol_srednja=("Solubility", "mean"),
    sol_std=("Solubility", "std"),
).sort_index()
 
print("Statistika po grupama pouzdanosti:")
print(group_stats.round(3).to_string())
 
# %% 4. Grupe pouzdanosti — grafici
 
order  = ["G1", "G2", "G3", "G4", "G5"]
colors = ["#7eb0d5", "#fd7f6f", "#b2e061", "#fd7f6f", "#b2e061"]
 
group_labels = {
    "G1": "G1\n(jedan izvor,\nnevalidovano)",
    "G2": "G2\n(2 izvora,\nneslaganje)",
    "G3": "G3\n(2 izvora,\nslaganje)",
    "G4": "G4\n(3+ izvora,\nneslaganje)",
    "G5": "G5\n(3+ izvora,\nslaganje ✓)",
}
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Levo: broj molekula po grupi
ax = axes[0]
cnts = [group_stats.loc[g, "broj"] if g in group_stats.index else 0 for g in order]
bars = ax.bar([group_labels[g] for g in order], cnts, color=colors, edgecolor="white")
for bar, v in zip(bars, cnts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(v), ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Broj molekula")
ax.set_title("Veličine grupa pouzdanosti")
 
# Desno: box plot rastvorljivosti po grupi
ax2 = axes[1]
group_order = [g for g in order if g in df["Group"].unique()]
sns.boxplot(data=df, x="Group", y="Solubility", order=group_order,
            palette=dict(zip(order, colors)), ax=ax2)
ax2.axhline(sol.mean(), color="crimson", linestyle="--", lw=1,
            label=f"ukupna srednja vred. {sol.mean():.2f}")
ax2.set_xlabel("Grupa pouzdanosti")
ax2.set_ylabel("log S  (mol/L)")
ax2.set_title("Raspodela rastvorljivosti po grupama pouzdanosti")
ax2.legend()
 
fig.tight_layout()
save(fig, "02_grupe_pouzdanosti.png")
 
# %% [markdown]
# ## 5. Analiza deskriptora
#
# Skup podataka sadrži 17 unapred izračunatih 2D deskriptora.
# Analiziramo njihove korelacije sa ciljnom promenljivom i međusobne korelacije.
# **MolLogP** (hidrofobnost) je očekivano najjači prediktor rastvorljivosti.
 
# %% 5. Analiza deskriptora — korelacije
 
desc_df = df[DESCRIPTOR_COLS + ["Solubility"]].copy()
corrs   = desc_df.corr()["Solubility"].drop("Solubility").sort_values()
 
print("Pearsonove korelacije deskriptora sa log S:")
print(corrs.round(3).to_string())
 
# %% 5. Analiza deskriptora — stubičasti grafikon korelacija
 
fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ["#fd7f6f" if c < 0 else "#b2e061" for c in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearsonov r sa log S")
ax.set_title("Korelacije deskriptora sa rastvorljivošću")
for i, (name, val) in enumerate(corrs.items()):
    ax.text(val + (0.01 if val >= 0 else -0.01), i,
            f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
fig.tight_layout()
save(fig, "03a_korelacije_deskriptora.png")
 
# %% 5. Analiza deskriptora — mapa korelacija (heatmap)
 
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(desc_df.corr(), dtype=bool))  # prikazujemo samo donji trougao
sns.heatmap(desc_df.corr(), mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
ax.set_title("Mapa korelacija deskriptora i ciljne promenljive")
fig.tight_layout()
save(fig, "03b_mapa_korelacija.png")
 
# %% 5. Analiza deskriptora — dijagrami rasipanja (top 6)
 
top6 = corrs.abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
 
for ax, feat in zip(axes.flat, top6):
    r = corrs[feat]
    ax.scatter(df[feat], df["Solubility"], alpha=0.15, s=8, color=PALETTE[0])
    # Linija linearne regresije
    m, b = np.polyfit(df[feat], df["Solubility"], 1)
    xs   = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(xs, m*xs + b, color="crimson", linewidth=1.5)
    ax.set_xlabel(feat)
    ax.set_ylabel("log S")
    ax.set_title(f"{feat}  (r = {r:.2f})")
 
fig.suptitle("Top 6 deskriptora u odnosu na rastvorljivost", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "03c_dijagrami_rasipanja.png")
 
# %% 5. Analiza deskriptora — raspodele svih deskriptora
 
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
axes_flat = axes.flat
 
for feat in DESCRIPTOR_COLS:
    ax = next(axes_flat)
    sns.histplot(df[feat], bins=40, kde=True, color=PALETTE[2], ax=ax, line_kws={"lw": 1.5})
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
 
# Sakrivamo neiskorišćene subplot-ove
for ax in axes_flat:
    ax.set_visible(False)
 
fig.suptitle("Raspodele deskriptora", fontsize=13, y=1.01)
fig.tight_layout()
save(fig, "03d_raspodele_deskriptora.png")
 
# %% [markdown]
# ## 6. Pregled podele podataka
#
# Podela **80 / 10 / 10** sa fiksiranim semenom `seed=42`.
# Ista podela koristi se za sva tri modela (MLP, GCN, GAT) —
# jedino tako je poređenje arhitektura kontrolisano i validno.
 
# %% 6. Pregled podele podataka
 
idx = np.arange(len(df))
idx_tmp, idx_test = train_test_split(idx, test_size=SPLIT_VAL,    random_state=SEED)
idx_tr,  idx_val  = train_test_split(idx_tmp, test_size=SPLIT_TEST, random_state=SEED)
 
splits = {
    "Trening"   : df.iloc[idx_tr],
    "Validacija": df.iloc[idx_val],
    "Test"      : df.iloc[idx_test],
}
 
print(f"{'Skup':<12} {'N':>6}  {'%':>5}  {'sred. logS':>10}  {'std logS':>9}  {'min':>7}  {'max':>7}")
for name, sub in splits.items():
    s = sub["Solubility"]
    print(f"{name:<12} {len(sub):>6}  {100*len(sub)/len(df):>4.1f}%  "
          f"{s.mean():>10.3f}  {s.std():>9.3f}  {s.min():>7.2f}  {s.max():>7.2f}")
 
# Provera: raspodele po podskupovima treba da budu slične
print("\nSastav grupa pouzdanosti po podskupovima (%):")
group_comp = pd.DataFrame({
    name: sub["Group"].value_counts(normalize=True).mul(100).round(1)
    for name, sub in splits.items()
}).reindex(["G1","G2","G3","G4","G5"]).fillna(0)
print(group_comp.to_string())
 
# %% 6. Pregled podele podataka — grafik
 
fig, ax = plt.subplots(figsize=(9, 5))
colors_split = {"Trening": PALETTE[0], "Validacija": PALETTE[1], "Test": PALETTE[2]}
 
for name, sub in splits.items():
    sns.kdeplot(sub["Solubility"], label=f"{name} (n={len(sub):,})",
                color=colors_split[name], linewidth=2, ax=ax)
 
ax.set_xlabel("log S  (mol/L)")
ax.set_ylabel("Gustina")
ax.set_title("Raspodela rastvorljivosti po podskupovima")
ax.legend()
fig.tight_layout()
save(fig, "04_raspodela_podskupova.png")
 
# %% [markdown]
# ## 7. Detekcija outlier-a
#
# Outlier-i definisani kao `|z| > 3` — više od 3 standardne devijacije od srednje vrednosti.
# Očekujemo jako nerastvorljive halogenovane ugljovodonike (PCB, dioksini, furani).
 
# %% 7. Detekcija outlier-a
 
z_scores = np.abs(stats.zscore(sol))
outliers = df[z_scores > 3][["Name", "SMILES", "Solubility", "Group"]].copy()
 
print(f"Broj outlier-a (|z| > 3): {len(outliers)}")
print(outliers.sort_values("Solubility").to_string())
 
# %% [markdown]
# ## 8. Završni izveštaj
 
# %% 8. Završni izveštaj
 
print(f"""
╔══════════════════════════════════════════════════════════╗
║         MolGraph - Rezime eksplorativne analize          ║
╚══════════════════════════════════════════════════════════╝
 
  Skup podataka (posle uklanjanja višekomponentnih)
  ───────────────────────────────────────────────────
  Ukupno molekula        : {len(df):,}
  Broj kolona            : {df.shape[1]}  ({len(DESCRIPTOR_COLS)} deskriptora)
  Nedostajuće vrednosti  : 0  ✓
  Duplirani SMILES       : 0  ✓
 
  Ciljna promenljiva  (log S, mol/L)
  ────────────────────────────────────
  Opseg        : {sol.min():.2f}  do  {sol.max():.2f}
  Srednja ± std: {sol.mean():.2f} ± {sol.std():.2f}
  Medijana     : {sol.median():.2f}
  Asimetrija   : {skew:.3f}
 
  Kategorije rastvorljivosti
  ────────────────────────────
  Visoko rastvorljivo (> 0)     : {(sol > 0).sum():4d}  ({100*(sol > 0).mean():.1f}%)
  Rastvorljivo (0 do -2)        : {((sol <= 0) & (sol > -2)).sum():4d}  ({100*((sol <= 0) & (sol > -2)).mean():.1f}%)
  Slabo rastvorljivo (-2 do -4) : {((sol <= -2) & (sol > -4)).sum():4d}  ({100*((sol <= -2) & (sol > -4)).mean():.1f}%)
  Nerastvorljivo (< -4)         : {(sol <= -4).sum():4d}  ({100*(sol <= -4).mean():.1f}%)
 
  Grupe pouzdanosti
  ──────────────────
  G1 (jedan izvor)           : {(df.Group=='G1').sum():4d}  ({100*(df.Group=='G1').mean():.1f}%)
  G2 (2 izvora, neslaganje)  : {(df.Group=='G2').sum():4d}  ({100*(df.Group=='G2').mean():.1f}%)
  G3 (2 izvora, slaganje)    : {(df.Group=='G3').sum():4d}  ({100*(df.Group=='G3').mean():.1f}%)
  G4 (3+ izvora, neslaganje) : {(df.Group=='G4').sum():4d}  ({100*(df.Group=='G4').mean():.1f}%)
  G5 (3+ izvora, slaganje ✓) : {(df.Group=='G5').sum():4d}  ({100*(df.Group=='G5').mean():.1f}%)
 
  Podela podataka (seme={SEED})
  ──────────────────────────────
  Trening    : {len(idx_tr):,}  ({100*len(idx_tr)/len(df):.1f}%)
  Validacija : {len(idx_val):,}  ({100*len(idx_val)/len(df):.1f}%)
  Test       : {len(idx_test):,}  ({100*len(idx_test)/len(df):.1f}%)
 
  Top 3 deskriptora po korelaciji sa log S
  ──────────────────────────────────────────
  {corrs.abs().sort_values(ascending=False).head(3).to_string()}
 
  Outlier-i (|z| > 3)        : {len(outliers)}
  Grafici sačuvani u         : {FIGURES_DIR}/
""")
