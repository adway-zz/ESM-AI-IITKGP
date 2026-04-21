import random
from pathlib import Path
import numpy as np
import torch 
import yaml
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download, snapshot_download


from PrithviWxC.model import PrithviWxC
from PrithviWxC.dataloaders.merra2 import Merra2Dataset,preproc
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time, os
from tqdm import tqdm

## configuring the device
torch.jit.enable_onednn_fusion(True)
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

## Defining the variables 

surface_vars = [
    "PRECTOT",
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M" ]

static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
levels = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
]
padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}

# Fine-tuning configuration
lead_times = [3]  # Predict multiple lead times
input_times = [-3, -6]  # Use multiple input timesteps

variable_names = surface_vars + [
    f'{var}_level_{level}' for var in vertical_vars for level in levels
]

time_range = ("2020-07-01T00:00:00", "2020-07-31T23:59:59")

surf_dir = Path("./data/merra-2")

snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    allow_patterns=[
        "merra-2/MERRA2_sfc_20200701.nc",
        "merra-2/MERRA2_sfc_20200702.nc",
        "merra-2/MERRA2_sfc_20200703.nc",
        "merra-2/MERRA2_sfc_20200704.nc",
        "merra-2/MERRA2_sfc_20200705.nc",
    ],
    local_dir="./data",
)

vert_dir = Path("./data/merra-2")

snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    allow_patterns=[
        "merra-2/MERRA_pres_20200701.nc",
        "merra-2/MERRA_pres_20200702.nc",
        "merra-2/MERRA_pres_20200703.nc",
        "merra-2/MERRA_pres_20200704.nc",
        "merra-2/MERRA_pres_20200705.nc",
    ],
    local_dir="./data",
)

surf_clim_dir = Path("./data/climatology")

snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    allow_patterns=[
        "climatology/climate_surface_doy183*.nc",
        "climatology/climate_surface_doy184*.nc",
        "climatology/climate_surface_doy185*.nc",
        "climatology/climate_surface_doy186*.nc",
        "climatology/climate_surface_doy187*.nc",
    ],
    local_dir="./data",
)

vert_clim_dir = Path("./data/climatology")

snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
    allow_patterns=[
        "climatology/climate_vertical_doy183*.nc",
        "climatology/climate_vertical_doy184*.nc",
        "climatology/climate_vertical_doy185*.nc",
        "climatology/climate_vertical_doy186*.nc",
        "climatology/climate_vertical_doy187*.nc",
    ],
    local_dir="./data",
)



positional_encoding = "fourier"


print(Merra2Dataset.valid_surface_vars)


dataset = Merra2Dataset(
    time_range=time_range,
    lead_times=lead_times,
    input_times=input_times,
    data_path_surface=surf_dir,
    data_path_vertical=vert_dir,
    climatology_path_surface=surf_clim_dir,
    climatology_path_vertical=vert_clim_dir,
    surface_vars=surface_vars,
    static_surface_vars=static_surface_vars,
    vertical_vars=vertical_vars,
    levels=levels,
    positional_encoding=positional_encoding,
)
assert len(dataset) > 0, "There doesn't seem to be any valid data."

data_dir = Path("climatology")

surf_in_scal_path  = data_dir / "musigma_surface.nc"
vert_in_scal_path  = data_dir / "musigma_vertical.nc"
surf_out_scal_path = data_dir / "anomaly_variance_surface.nc"
vert_out_scal_path = data_dir / "anomaly_variance_vertical.nc"

in_mu, in_sig = input_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_in_scal_path,
    vert_in_scal_path,
)

output_sig = output_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_out_scal_path,
    vert_out_scal_path,
)

static_mu, static_sig = static_input_scalers(
    surf_in_scal_path,
    static_surface_vars,
)

print("✅ Scalers loaded successfully!")

residual = "climate"
masking_mode = "global"
encoder_shifting = True
decoder_shifting = True
masking_ratio = 0.0


## defining the base model 
data_dir = Path("")

with open(data_dir / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

config["params"]["in_channels"] = 161

model = PrithviWxC(
    in_channels=config["params"]["in_channels"],
    input_size_time=config["params"]["input_size_time"],
    in_channels_static=config["params"]["in_channels_static"],
    input_scalers_mu=in_mu,
    input_scalers_sigma=in_sig,
    input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
    static_input_scalers_mu=static_mu,
    static_input_scalers_sigma=static_sig,
    static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
    output_scalers=output_sig**0.5,
    n_lats_px=config["params"]["n_lats_px"],
    n_lons_px=config["params"]["n_lons_px"],
    patch_size_px=config["params"]["patch_size_px"],
    mask_unit_size_px=config["params"]["mask_unit_size_px"],
    mask_ratio_inputs=masking_ratio,
    mask_ratio_targets=0.0,
    embed_dim=config["params"]["embed_dim"],
    n_blocks_encoder=config["params"]["n_blocks_encoder"],
    n_blocks_decoder=config["params"]["n_blocks_decoder"],
    mlp_multiplier=config["params"]["mlp_multiplier"],
    n_heads=config["params"]["n_heads"],
    dropout=config["params"]["dropout"],
    drop_path=config["params"]["drop_path"],
    parameter_dropout=config["params"]["parameter_dropout"],
    residual=residual,
    masking_mode=masking_mode,
    encoder_shifting=encoder_shifting,
    decoder_shifting=decoder_shifting,
    positional_encoding=positional_encoding,
    checkpoint_encoder=[],
    checkpoint_decoder=[],
)


weights_path = Path("prithvi.wxc.2300m.v1.pt")

# Load weights into model
state_dict = torch.load(weights_path, weights_only=False)
if "model_state" in state_dict:
    state_dict = state_dict["model_state"]

# Load only matching keys
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

# Update model dict
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print("✅ Model loaded successfully!")

######################################################

class SimplePrecipitationDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, surface_vars, prectot_idx, max_samples=None):
        print("\n=== Initializing SimplePrecipitationDataset ===")
        print(f"Base dataset length: {len(base_dataset)}")
        print(f"Surface vars: {surface_vars}")
        print(f"PRECTOT index: {prectot_idx}")
        
        self.base_dataset = base_dataset
        self.surface_vars = surface_vars
        self.prectot_idx = prectot_idx
        
        # Just take indices (no validation loop)
        self.indices = list(range(len(base_dataset)))
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
        
        print(f"Using {len(self.indices)} samples (no pre-validation)")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.base_dataset[actual_idx]
        
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {actual_idx} is not a dict")

        if "sur_tars" not in sample or not torch.is_tensor(sample["sur_tars"]):
            raise ValueError(f"Sample {actual_idx} missing or invalid `sur_tars`")

        sur_tars = sample['sur_tars']  # Shape: [channels, time, height, width]
        if sur_tars.shape[0] <= self.prectot_idx:
            raise ValueError(f"Sample {actual_idx} has too few channels, expected PRECTOT at {self.prectot_idx}")
        
        prectot_target = sur_tars[self.prectot_idx, 0, :, :].clone()  # [height, width]
        
        # Build modified sample
        modified_sample = {}
        for key, value in sample.items():
            if key in ['sur_vals', 'sur_tars', 'sur_climate'] and torch.is_tensor(value):
                modified_value = value.clone()
                if modified_value.shape[0] > self.prectot_idx:
                    modified_value[self.prectot_idx] = 0.0  # Zero-out PRECTOT channel
                modified_sample[key] = modified_value
            else:
                modified_sample[key] = value
        
        return {"sample": modified_sample, "target": prectot_target}

# FIXED: Simple collate function
def simple_collate_fn(batch):
    """Simple collate function using PrithviWxC preprocessing"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # Collect samples and targets
    samples_list = [item['sample'] for item in batch]
    targets_list = [item['target'] for item in batch]
    
    try:
        # Use the official preproc function
        processed_batch = preproc(samples_list, padding)
        targets = torch.stack(targets_list)
        
        return {'inputs': processed_batch, 'targets': targets}
        
    except Exception as e:
        print(f"Error in simple_collate_fn: {e}")
        import traceback
        traceback.print_exc()
        return None

class FixedPrecipitationModel(torch.nn.Module):
    def __init__(self, base_model, prectot_idx):
        super().__init__()
        self.base_model = base_model
        self.prectot_idx = prectot_idx
        
        # UNFREEZE MORE LAYERS - based on debug showing minimal learning
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.base_model.named_parameters():
            # Unfreeze decoder layers + final layers for better capacity
            if any(layer in name.lower() for layer in ['unembed', 'norm', 'final', 'decoder']):
                param.requires_grad = True
                trainable_count += param.numel()
                print(f"Unfrozen: {name}")
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        print(f"Frozen parameters: {frozen_count:,}")
        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Trainable ratio: {trainable_count/(frozen_count+trainable_count):.4f}")
    
    def forward(self, inputs):
        full_output = self.base_model(inputs)
        prectot_output = full_output[:, self.prectot_idx, :, :]
        return prectot_output

class ScaledLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        # Convert kg/m²/s to mm/day for better numerical stability
        self.scale_factor = 86400.0  # seconds per day
    
    def forward(self, pred, target):
        # Handle shape mismatch
        if pred.shape != target.shape:
            if len(target.shape) == 3 and len(pred.shape) == 3:
                target = torch.nn.functional.interpolate(
                    target.unsqueeze(1), size=pred.shape[-2:], 
                    mode='bilinear', align_corners=False
                ).squeeze(1)
        
        # Scale both to mm/day for numerical stability
        pred_scaled = pred * self.scale_factor
        target_scaled = target * self.scale_factor
        
        # MSE in mm/day units
        loss = self.mse(pred_scaled, target_scaled)
        
        return loss, {
            'mse_mmday': loss.item(),
            'pred_mean_mmday': pred_scaled.mean().item(),
            'target_mean_mmday': target_scaled.mean().item(),
            'pred_std_mmday': pred_scaled.std().item(),
            'target_std_mmday': target_scaled.std().item(),
        }


# Create the precipitation dataset
print("\n=== Creating Precipitation Dataset ===")
prectot_idx = surface_vars.index('PRECTOT')
print(f"PRECTOT index in surface_vars: {prectot_idx}")

# Limit samples for testing (remove max_samples=50 for full dataset)
precip_dataset = SimplePrecipitationDataset(
    dataset, surface_vars, prectot_idx, max_samples=100  # Start small for testing
)

print(f"\n=== Splitting dataset (80% train, 20% val) ===")

train_size = int(0.8 * len(precip_dataset))
val_size = int(0.2 * len(precip_dataset))

train_dataset, val_dataset, _ = torch.utils.data.random_split(
    precip_dataset, [train_size, val_size, len(precip_dataset) - train_size - val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")


# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,  # Small batch size
    shuffle=True,
    collate_fn=simple_collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=simple_collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# Create the model
print("\n=== Creating Model ===")
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

simple_precip_model = FixedPrecipitationModel(model, prectot_idx)
simple_precip_model = simple_precip_model.to(device)


# Setup training
criterion = ScaledLoss()
optimizer = torch.optim.AdamW(
    [p for p in simple_precip_model.parameters() if p.requires_grad],
    lr=1e-4,  # Higher LR since gradients were tiny
    weight_decay=1e-6,
    betas=(0.9, 0.95)  # Better for fine-tuning
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)


def get_indian_region_mask(lat_coords, lon_coords, lat_range=(6.0, 37.0), lon_range=(68.0, 97.0)):
    """
    Create a boolean mask for the Indian region
    
    Args:
        lat_coords: Array of latitude coordinates
        lon_coords: Array of longitude coordinates  
        lat_range: Tuple of (min_lat, max_lat) for India
        lon_range: Tuple of (min_lon, max_lon) for India
    
    Returns:
        Boolean mask array for Indian region
    """
    # Convert to numpy arrays if they aren't already
    lat_coords = np.array(lat_coords)
    lon_coords = np.array(lon_coords)
    
    # Create meshgrid for 2D coordinates
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Create mask for Indian region
    lat_mask = (lat_grid >= lat_range[0]) & (lat_grid <= lat_range[1])
    lon_mask = (lon_grid >= lon_range[0]) & (lon_grid <= lon_range[1])
    
    indian_mask = lat_mask & lon_mask
    
    return indian_mask

def compute_metrics_with_indian_region(y_true, y_pred, indian_mask=None, scale_factor=86400.0):
    """
    Compute metrics for both global and Indian region
    
    Args:
        y_true: Ground truth values [batch, height, width]
        y_pred: Predicted values [batch, height, width]
        indian_mask: Boolean mask for Indian region [height, width]
        scale_factor: Conversion factor (default: kg/m²/s to mm/day)
    
    Returns:
        Dictionary with global and Indian region metrics
    """
    # Handle shape mismatches by cropping to common dimensions
    if y_true.shape != y_pred.shape:
        print(f"Shape mismatch in metrics: true {y_true.shape} vs pred {y_pred.shape}")
        
        # Get minimum dimensions
        min_samples = min(y_true.shape[0], y_pred.shape[0])
        if len(y_true.shape) > 2:  # spatial dimensions exist
            min_h = min(y_true.shape[-2], y_pred.shape[-2])
            min_w = min(y_true.shape[-1], y_pred.shape[-1])
            
            # Crop both to same size
            y_true = y_true[:min_samples, :min_h, :min_w]
            y_pred = y_pred[:min_samples, :min_h, :min_w]
            
            # Also crop the mask if provided
            if indian_mask is not None:
                indian_mask = indian_mask[:min_h, :min_w]
        else:
            y_true = y_true[:min_samples]
            y_pred = y_pred[:min_samples]
        
        print(f"Cropped to common shape: {y_true.shape}")
    
    # Convert to mm/day
    y_true_scaled = y_true * scale_factor
    y_pred_scaled = y_pred * scale_factor
    
    def calculate_metrics(true_vals, pred_vals, region_name=""):
        """Helper function to calculate metrics for a region"""
        true_flat = true_vals.flatten()
        pred_flat = pred_vals.flatten()
        
        # Ensure same length after flattening
        min_len = min(len(true_flat), len(pred_flat))
        true_flat = true_flat[:min_len]
        pred_flat = pred_flat[:min_len]
        
        mse = np.mean((true_flat - pred_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_flat - pred_flat))
        
        # Better R² calculation
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Correlation coefficient
        if np.std(true_flat) > 1e-8 and np.std(pred_flat) > 1e-8:
            corr = np.corrcoef(true_flat, pred_flat)[0, 1]
        else:
            corr = 0.0
        
        return {
            f'mse_mmday{region_name}': mse,
            f'rmse_mmday{region_name}': rmse, 
            f'mae_mmday{region_name}': mae,
            f'r2{region_name}': r2,
            f'correlation{region_name}': corr,
            f'target_mean_mmday{region_name}': np.mean(true_flat),
            f'pred_mean_mmday{region_name}': np.mean(pred_flat),
            f'target_std_mmday{region_name}': np.std(true_flat),
            f'pred_std_mmday{region_name}': np.std(pred_flat),
            f'n_points{region_name}': len(true_flat)
        }
    
    # Global metrics
    metrics = calculate_metrics(y_true_scaled, y_pred_scaled, "_global")
    
    # Indian region metrics
    if indian_mask is not None:
        # Apply mask to extract Indian region data
        indian_true = y_true_scaled[:, indian_mask]  # [batch, indian_pixels]
        indian_pred = y_pred_scaled[:, indian_mask]  # [batch, indian_pixels]
        
        indian_metrics = calculate_metrics(indian_true, indian_pred, "_indian")
        metrics.update(indian_metrics)
    
    return metrics


lat_coords = np.arange(-90, 90.5, 0.5)  # MERRA-2 standard lat grid
lon_coords = np.arange(-180, 180, 0.625)  # MERRA-2 standard lon grid

# Create Indian region mask
indian_mask = get_indian_region_mask(lat_coords, lon_coords)
print(f"Indian region mask shape: {indian_mask.shape}")
print(f"Indian region covers {indian_mask.sum()} grid points ({indian_mask.sum()/indian_mask.size*100:.2f}% of total grid)")


print("\n" + "="*50)
print("STARTING FIXED TRAINING WITH INDIAN REGION METRICS")
print("="*50)

num_epochs = 20
best_val_loss = float('inf')
best_val_loss_indian = float('inf')
train_losses = []
val_losses = []
train_losses_indian = []
val_losses_indian = []

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training
    simple_precip_model.train()
    epoch_train_loss = 0.0
    train_batches = 0
    all_train_preds = []
    all_train_targets = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
        if batch is None:
            continue
        
        try:
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch['inputs'].items()}
            targets = batch['targets'].to(device)
            
            # Forward pass
            predictions = simple_precip_model(inputs)
            loss, metrics = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in simple_precip_model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
            
            # Store for metrics
            all_train_preds.append(predictions.detach().cpu().numpy())
            all_train_targets.append(targets.detach().cpu().numpy())
            
            # Debug first batch
            if batch_idx == 0:
                print(f"  Batch 0: Loss={loss.item():.2f}, Grad_norm={grad_norm:.6f}")
                print(f"  Metrics: {metrics}")
        
        except Exception as e:
            print(f"Training error: {e}")
            continue
    
    # Validation
    simple_precip_model.eval()
    epoch_val_loss = 0.0
    val_batches = 0
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            if batch is None:
                continue
            
            try:
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch['inputs'].items()}
                targets = batch['targets'].to(device)
                
                predictions = simple_precip_model(inputs)
                loss, _ = criterion(predictions, targets)
                
                epoch_val_loss += loss.item()
                val_batches += 1
                
                all_val_preds.append(predictions.cpu().numpy())
                all_val_targets.append(targets.cpu().numpy())
                
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    # Calculate metrics
    avg_train_loss = epoch_train_loss / max(train_batches, 1)
    avg_val_loss = epoch_val_loss / max(val_batches, 1)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Calculate detailed metrics with Indian region
    train_metrics = {}
    val_metrics = {}
    
    if all_train_preds and all_train_targets:
        train_preds = np.concatenate(all_train_preds, axis=0)
        train_targets = np.concatenate(all_train_targets, axis=0)
        train_metrics = compute_metrics_with_indian_region(train_targets, train_preds, indian_mask)
    
    if all_val_preds and all_val_targets:
        val_preds = np.concatenate(all_val_preds, axis=0)
        val_targets = np.concatenate(all_val_targets, axis=0)
        val_metrics = compute_metrics_with_indian_region(val_targets, val_preds, indian_mask)
    
    # Store Indian region specific losses
    if 'mse_mmday_indian' in train_metrics:
        train_losses_indian.append(train_metrics['mse_mmday_indian'])
    if 'mse_mmday_indian' in val_metrics:
        val_losses_indian.append(val_metrics['mse_mmday_indian'])
    
    # Learning rate scheduling
    scheduler.step()
    
    elapsed = time.time() - start_time
    
    # Enhanced printing with Indian region metrics
    print(f"\nEpoch {epoch+1}/{num_epochs} - {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"GLOBAL METRICS:")
    print(f"Train Loss: {avg_train_loss:.2f} mm²/day²")
    print(f"Val Loss:   {avg_val_loss:.2f} mm²/day²")
    
    if train_metrics and val_metrics:
        print(f"Train RMSE: {train_metrics.get('rmse_mmday_global', 0):.3f} mm/day, R²: {train_metrics.get('r2_global', 0):.4f}")
        print(f"Val RMSE:   {val_metrics.get('rmse_mmday_global', 0):.3f} mm/day, R²: {val_metrics.get('r2_global', 0):.4f}")
        
        print(f"\nINDIAN REGION METRICS:")
        if 'mse_mmday_indian' in train_metrics and 'mse_mmday_indian' in val_metrics:
            print(f"Train Loss: {train_metrics['mse_mmday_indian']:.2f} mm²/day²")
            print(f"Val Loss:   {val_metrics['mse_mmday_indian']:.2f} mm²/day²")
            print(f"Train RMSE: {train_metrics['rmse_mmday_indian']:.3f} mm/day, R²: {train_metrics['r2_indian']:.4f}")
            print(f"Val RMSE:   {val_metrics['rmse_mmday_indian']:.3f} mm/day, R²: {val_metrics['r2_indian']:.4f}")
            print(f"Correlation: {val_metrics['correlation_indian']:.4f}")
            print(f"Grid points: {val_metrics['n_points_indian']:,}")
        else:
            print("Indian region metrics not available (check mask dimensions)")
    
    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*60}")
    
    # Save best model (based on global validation loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        # Also track best Indian region loss
        if 'mse_mmday_indian' in val_metrics:
            best_val_loss_indian = val_metrics['mse_mmday_indian']
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': simple_precip_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'indian_mask': indian_mask,
        }, 'best_fixed_precipitation_model.pt')
        
        print(f"→ Best model saved! (Global Val Loss: {avg_val_loss:.2f})")
        if 'mse_mmday_indian' in val_metrics:
            print(f"  Indian Region Val Loss: {val_metrics['mse_mmday_indian']:.2f}")

print(f"\nTraining completed!")
print(f"Best global validation loss: {best_val_loss:.2f} mm²/day²")
if best_val_loss_indian != float('inf'):
    print(f"Best Indian region validation loss: {best_val_loss_indian:.2f} mm²/day²")
