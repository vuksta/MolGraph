# MolGraph
 
Comparing MLP, GCN and GAT models for molecular solubility prediction.

Uni Project for "MITNOP" by Vuk Stamenković (IN-31/2023) and Lazar Stojšić (IN-41/2023)
 
---
 
## What this is
 
Given only a molecule's SMILES string, the models predict its aqueous solubility (log S in mol/L) which is a property that matters a lot in drug discovery and chemical engineering.
 
We train and compare three architectures under identical conditions to see whether explicitly encoding molecular topology actually helps:
 
- **MLP** — Morgan fingerprints (radius 2, 2048 bits), no topology
- **GCN** — molecule as a graph, edges treated equally
- **GAT** — molecule as a graph, edge weights learned via attention
  
The same dataset, same 80/10/10 split (seed=42), same Adam optimizer, same early stopping. The only thing that changes is the architecture.
 
---
 
## Dataset
 
**AqSolDB**: 9,982 unique compounds with experimentally measured solubility values, assembled from nine public sources by the AMD research group. Available on Harvard Dataverse: [doi:10.7910/DVN/OVHAW8](https://doi.org/10.7910/DVN/OVHAW8).
 
Each row has a SMILES string, a measured log S value, a reliability flag, and precomputed 2D descriptors. Target values range roughly from −12 (insoluble) to +2 (highly soluble).
 
---
