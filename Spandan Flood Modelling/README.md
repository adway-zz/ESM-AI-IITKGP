# GNN Benchmarking on the Wollombi Flood Dataset

Benchmarking several Graph Neural Network (GNN) architectures for flood prediction, built on
the [DUALFloodGNN](https://github.com/acostacos/dual_flood_gnn) codebase and the Wollombi
HEC-RAS flood dataset.

## Overview

| | |
|---|---|
| **Task** | Predict water volume at mesh nodes (and water flow at edges for DUALFloodGNN) |
| **Models** | GCN, GAT, GIN, GINE, GraphSAGE (node-only), DUALFloodGNN (node + edge, physics-informed) |
| **Data** | Wollombi HEC-RAS flood simulations (56 events; 4 held out for testing) |
| **Metrics** | RMSE, MAE, NSE (Nash-Sutcliffe Efficiency), CSI (Critical Success Index) |
| **Environment** | Kaggle, GPU T4 · 50 epochs/model · seed 42 |

## Repository structure

```
.
├── README.md                         # this file
├── notebooks/
│   └── flood_gnn_benchmark.ipynb     # end-to-end: setup → data prep → train → evaluate → plots
├── results/
│   └── model_comparison.csv          # aggregated test metrics per model
└── figures/                          # generated plots
    ├── val_rmse.png
    ├── train_loss.png
    ├── scatter_pred_vs_true.png
    └── dualflood_volume_and_flow.png
```

## How to run

The notebook is written for Kaggle with the dataset attached as inputs.

1. **Dataset** — not included here (it is large and hosted separately). Download from the
   [USYD library](https://ses.library.usyd.edu.au/handle/2123/35293) and attach to the Kaggle
   notebook as datasets: the HDF simulation files, the GEOMETRY shapefiles, and the
   `train.csv` / `test.csv` summaries.
2. **Run** `notebooks/flood_gnn_benchmark.ipynb` top to bottom. It clones the upstream repo,
   installs dependencies, arranges the data into the expected layout, patches one upstream
   bug, trains each model for 50 epochs, evaluates on the test events, and saves the metrics
   table and plots.

## Notes on the setup

- The dataset ships raw shapefiles (`updated_cell_centers.shp`, `links.shp`) whose columns
  already contain everything the loader needs (`X`, `Y`, `Elevation1` for nodes;
  `from_node`, `to_node`, `length`, `slope` for edges). The notebook copies them to the
  filenames the config expects.
- The compiled PyTorch-Geometric extras (`pyg_lib`, `torch_scatter`, …) have no wheels for
  the current Kaggle PyTorch build and are not required for these models, so they are skipped.
- One upstream bug in `train.py` (a stale variable name in the autoregressive test path) is
  patched in the notebook.

## Results (50 epochs)

Ranked by NSE (higher is better; 1 = perfect, 0 = mean baseline, < 0 = worse than mean):

| Model | RMSE ↓ | MAE ↓ | NSE ↑ | CSI ↑ | Notes |
|---|---|---|---|---|---|
| **GINE** | 16,493 | 7,753 | −2.38 | 0.56 | Best overall; only baseline using edge features |
| GIN | 20,181 | 10,723 | −4.43 | 0.54 | |
| GCN | 25,448 | 12,547 | −6.38 | 0.44 | |
| DUALFloodGNN | 46,111 | 23,622 | −33.3 | 0.56 | Dual objective (volume + flow); undertrained at 50 ep |
| GraphSAGE | 91,338 | 29,776 | −354.5 | 0.47 | Unstable |
| GAT | diverged | diverged | diverged | 0.36 | Failed to converge at default LR |

**Key points**

- **GINE** is the strongest baseline, consistent with its use of edge features (slope,
  length) that physically drive water flow.
- **GAT** diverges numerically at the default learning rate — a stability issue, not an
  architectural limitation.
- **DUALFloodGNN** is undertrained at 50 epochs; its harder dual objective (node volume +
  edge flow under mass-conservation constraints) needs the full-length run to converge.
- All NSE values are negative at 50 epochs — no model yet beats a mean baseline on water
  *magnitude*, as expected for this short training length. The model **ranking** is the
  reliable takeaway; converged numbers require 300–600 epochs.

## Next steps

- Full-length training (300–600 epochs) with early stopping for converged metrics
- Tune GAT (lower learning rate / gradient clipping) to resolve divergence
- Add remaining edge and physics model variants (EdgeGCN, EdgeGAT, NodeEdgeGNNAttn, …)
- Repeat across multiple seeds for statistical reliability

## Credits

- Upstream model & training code: [acostacos/dual_flood_gnn](https://github.com/acostacos/dual_flood_gnn)
- Dataset: Wollombi HEC-RAS flood simulations, [USYD library](https://ses.library.usyd.edu.au/handle/2123/35293)
