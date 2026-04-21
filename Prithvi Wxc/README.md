# Prithvi WxC — experiments and fine-tuning

This workspace contains code and notebooks for working with **Prithvi-WxC**, IBM/NASA’s weather and climate foundation model, using **MERRA-2** inputs and Hugging Face assets.

## Contents

| Path | Purpose |
|------|--------|
| `PrithviWxC/` | Core library: `PrithviWxC` model, MERRA-2 dataloaders, rollout utilities, download helpers, and variable definitions. |
| `config.yaml` | Model architecture configuration (channels, grid, ViT blocks, attention, etc.). |
| `finetune.py` | End-to-end script: data download, dataset construction, loading pretrained weights, fine-tuning for precipitation (`PRECTOT`), evaluation (including Indian-region mask), and saving `best_fixed_precipitation_model.pt`. |
| `pipeline-setup.ipynb` | Pipeline setup and exploration. |
| `downscaling.ipynb` | Downscaling-related experiments. |
| `trying_the_maunet.ipynb` | Additional architecture experiments. |
| `Dataset/` | Dataset notes or paths (e.g. `location.md`). |

## Model and weights

- Architecture: `PrithviWxC` (defined in `PrithviWxC/model.py`)
- Configuration: `config.yaml` (e.g. `in_channels: 160`; may be overridden in `finetune.py`)
- Pretrained weights: `prithvi.wxc.2300m.v1.pt` (expected in project root; update path if needed)
- Example dataset: `ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M` (downloaded via `snapshot_download` in `finetune.py`)

## Data layout (used by `finetune.py`)

- Stored under `./data/`:
  - `data/merra-2/` — MERRA-2 NetCDF inputs  
  - `data/climatology/` — climatology statistics  

- Additional required files (in `climatology/` at repo root):
  - `musigma_surface.nc`
  - `musigma_vertical.nc`
  - anomaly variance files  

Ensure these files are available or update paths in the script accordingly.