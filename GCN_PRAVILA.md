# Model 2 — GCN: plan i pravila

Ovaj dokument služi dvostruko:

- **kao ugovor** za onoga ko piše GCN granu (šta sme, a šta ne sme da menja),
- **kao objašnjenje** za onoga ko čita projekat (zašto je GCN napravljen baš ovako).

> Model 1 (MLP nad Morganovim otiscima) je **već završen** — vidi
> [`notebooks/MLP_implementiran.py`](notebooks/MLP_implementiran.py). GCN se **ne** pravi
> od nule: nasleđuje ceo tok obuke od MLP-a i menja samo dva dela. Detalji ispod.

---

## 1. Zašto GCN uopšte postoji

Projekat poredi **tri arhitekture** na istom zadatku (regresija rastvorljivosti, log S):

| Model | Reprezentacija | Topologija molekula | Uloga |
|-------|----------------|---------------------|-------|
| MLP   | Morganovi otisci (2048 bita) | ❌ izgubljena | referentna tačka |
| **GCN** | **molekul kao graf** | ✅ koristi se | **ovaj dokument** |
| GAT   | graf + mehanizam pažnje | ✅ + težine veza | sledeći korak |

Cilj poređenja: razlike u rezultatima smeju da potiču **isključivo iz arhitekture**.
Zato sve ostalo (podaci, podela, optimizator, kriterijum zaustavljanja) mora da ostane
**identično** kao kod MLP-a.

---

## 2. Nepromenljivi ugovor (ne sme da se menja)

GCN **mora** da koristi iste ove delove kao MLP, bez ijedne izmene u logici ili
hiperparametrima. Ako se ovde nešto promeni, poređenje tri modela postaje nevažeće.

- **Čišćenje podataka — jednom, u `loadData`.** Izbacuju se SMILES zapisi sa tačkom
  (`.` razdvaja soli / smeše / kontrajone) i oni koje RDKit ne može da parsira.
  Rezultat: **8.882 molekula**. Sva tri modela treniraju na **istom** ovom skupu.
- **Podela 80 / 10 / 10**, fiksirano seme **`seed = 42`** (`splitData`). Iste molekule
  u trening / validaciji / testu kao kod MLP-a.
- **Protokol obuke:** optimizator **Adam**, funkcija greške **MSE**, **rano
  zaustavljanje** sa strpljenjem **patience = 20**, vraćanje najboljih težina.
- **Metrike:** **R²** i **RMSE** na testnom skupu (`score`).

> Pravilo: kopiraj hiperparametre iz MLP-a, ne „doteruj“ ih za GCN.

---

## 3. Šta se menja za GCN (i samo to)

Od devet funkcija MLP grane, **sedam** ostaje konceptualno isto. Menjaju se dve stvari
plus način pakovanja podataka u grupe (batch):

| Deo | MLP | GCN |
|-----|-----|-----|
| Featurizacija | `smileToFingerprint` + `buildDataSet` → vektor 2048 | **`smilesToGraph`** → graf (PyG `Data`) |
| Model | `SolubilityMLP` | **`SolubilityGCN`** |
| Pakovanje u grupe | `TensorDataset` + običan `DataLoader` | **`torch_geometric.loader.DataLoader`** (slaže više malih grafova u jedan) |

Sve ostalo — `loadData`, `splitData`, petlja obuke, rano zaustavljanje, računanje
R²/RMSE — zadržava **isti protokol i iste hiperparametre**. (Čitanje grupe unutar
petlje je tehnički drugačije jer se grafovi pakuju drugačije od vektora, ali
*pravila* obuke se ne diraju.)

---

## 4. Molekul kao graf

`smilesToGraph` pretvara jedan SMILES u objekat `torch_geometric.data.Data` sa:

- **`x`** — matrica osobina čvorova (atoma), dimenzije `[broj_atoma, broj_osobina]`
- **`edge_index`** — spisak veza, **svaka veza u oba smera** (graf je neusmeren)
- **`edge_attr`** — osobine veza
- **`y`** — ciljna vrednost (log S) za ceo molekul

### Osobine čvorova (atoma) — prema specifikaciji
- **atomski broj** → **one-hot** (vidi tačku 5; ovo je ključno)
- valenca
- aromatičnost (0/1)
- broj vodonikovih atoma
- naelektrisanje

### Osobine veza (atoma) — prema specifikaciji
- tip veze (jednostruka / dvostruka / trostruka / aromatična)
- aromatičnost (0/1)

> **Napomena za `edge_attr`:** čisti GCN **ne koristi** osobine veza — on usrednjava
> susede ravnopravno, „bez razlikovanja važnosti pojedinih veza“. Ipak ih
> **gradimo i čuvamo odmah**, jer ih **GAT (Model 3) koristi**. Pišemo `smilesToGraph`
> jednom, kompletno, da se kasnije ne prepravlja.

---

## 5. Ključna lekcija: atomski broj ide kao ONE-HOT

Ovo je najvažniji praktičan detalj cele GCN grane.

Ako se atomski broj prosledi kao **sirov broj** (ugljenik = 6, kiseonik = 8,
hlor = 17 …), te velike vrednosti **nadjačaju** male osobine (valenca, naelektrisanje
u opsegu 0–4) i mreža slabo uči — u praksi **podučenje (underfitting), R² ≈ 0,53**.

Rešenje: atomski broj se tretira kao **kategorija, a ne kao veličina** — one-hot
kodiranje (npr. C, N, O, F, … svaki kao zasebna pozicija). Time se R² popne na
**≈ 0,76**.

> Isto pravilo važi za svaku neograničenu osobinu: ili je one-hot, ili je skaliraj.
> Nikad sirov atomski broj kao ulaz.

---

## 6. Arhitektura `SolubilityGCN`

Proverena, jednostavna postavka:

```
3 × GCNConv  (svaki atom usrednjava osobine suseda)
      ↓
global_mean_pool   (atomi → jedan vektor po molekulu)
      ↓
mali MLP „head“    → 1 broj (predviđeni log S)
```

---

## 7. Očekivani rezultat

- Redosled iz hipoteze: **GCN > MLP** (graf koristi topologiju koju otisak gubi).
- Orijentir sa istog, očišćenog skupa: **GCN R² ≈ 0,75**, RMSE ≈ 1,1
  (MLP je oko R² ≈ 0,70). Tačni brojevi zavise od okruženja — **svi modeli moraju
  biti mereni u istom okruženju** da bi poređenje bilo pošteno.

---

## 8. Interpretabilnost (bitno za razumevanje granica GCN-a)

- **GCN nije interpretabilan po pojedinačnoj vezi** — i to je *namerno*. On sve
  susede usrednjava ravnopravno, pa nema „težine“ koje bismo mogli da pročitamo.
  Ovo je tačno ono što specifikacija opisuje: „bez razlikovanja važnosti pojedinih
  veza“.
- Interpretabilnost (vizualizacija težina pažnje na hidrofilnim grupama) je posao
  **Modela 3 (GAT)**. GCN ovde služi kao **kontrola bez pažnje** — referentna tačka
  naspram koje se vidi šta mehanizam pažnje stvarno donosi.

---

## 9. Gde kod ide u projektu

- Razvoj i testovi: `notebooks/` (npr. `GCN_implementiran.py`), paralelno sa
  `MLP_implementiran.py`.
- Kada proradi i ustali se, čista verzija se preseljava u `src/`
  (`data.py`, `model.py`, `train.py`, `evaluate.py`).
- Težine se čuvaju u `models/` (ignorisano u git-u).
- Grafici u `figures/`.

---

## 10. Kontrolna lista pre nego što se GCN proglasi gotovim

- [ ] `loadData` čišćenje neizmenjeno → **8.882** molekula
- [ ] `splitData` sa `seed = 42`, ista podela kao MLP
- [ ] atomski broj kao **one-hot** (ne sirov broj)
- [ ] `edge_attr` se gradi (iako ga GCN ne koristi) — spreman za GAT
- [ ] Adam + MSE + rano zaustavljanje `patience = 20`
- [ ] R² i RMSE na testu, izmereni u **istom okruženju** kao MLP
- [ ] rezultat zabeležen radi poređenja MLP / GCN / GAT
