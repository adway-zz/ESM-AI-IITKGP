## 📊 Dataset Description

The dataset available at:
👉 https://huggingface.co/datasets/ajaypaliwal/Prithvi-MTP

contains a collection of meteorological and rainfall data used for training and evaluating precipitation forecasting and downscaling models.

### 🔹 Overview
- Domain: Weather and climate forecasting  
- Focus: Rainfall prediction and downscaling over the Indian region  
- Data type: NetCDF (`.nc`) files and model checkpoints (`.pt`)  
- Usage: Training, fine-tuning, and evaluation of deep learning models (e.g., Prithvi-WxC, U-Net)

### 🔹 Contents

The dataset includes:

- **Rainfall Data**
  - `rainfall_2020_01degree_combined.nc`
  - High-resolution precipitation data (0.1° grid)

- **ERA5 Reanalysis Data**
  - `data_stream-oper_stepType-accum.nc` (accumulated variables)
  - `data_stream-oper_stepType-instant.nc` (instantaneous variables)
  - Contains atmospheric variables such as temperature, pressure, wind, and humidity

- **Model Checkpoints**
  - `best_unet_rainfall.pt` — Downscaling model
  - `best_fixed_precipitation_model.pt` — Fine-tuned Prithvi-WxC model
  - `prithvi.wxc.2300m.v1.pt` — Pretrained foundation model weights

### 🔹 Data Characteristics

- Spatial resolution:
  - Coarse: ~0.5°
  - High-resolution: ~0.1°
- Temporal resolution:
  - 3-hourly intervals (aligned for forecasting tasks)
- Format:
  - NetCDF for climate data
  - PyTorch `.pt` files for model weights

### 🔹 Use Cases

This dataset can be used for:

- Precipitation forecasting  
- Downscaling coarse weather data to high resolution  
- Fine-tuning foundation models like Prithvi-WxC  
- Evaluating model performance across different lead times  
- Regional climate analysis (especially over India)

### 🔹 Notes

- Data is structured for compatibility with deep learning pipelines  
- Suitable for both global and regional experiments  
- Large file sizes may require efficient data loading strategies  

---