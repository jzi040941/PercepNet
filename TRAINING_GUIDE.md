# Enhanced PercepNet Training Setup Guide

## Quick Start for Students

This guide provides everything needed to run PercepNet training with enhanced diagnostics and W&B logging.

## Prerequisites

### 1. Install Dependencies
```bash
# Install required packages
pip install wandb torch numpy h5py tqdm matplotlib pyyaml tensorboardx

# Login to W&B (first time only)
wandb login
```

### 2. Prepare Training Data

You need training data in the correct format. Choose one of these options:

#### Option A: Use existing HDF5 files
```bash
# If you have training.h5 files, they should be ready to use
ls training.h5  # Check if exists
```

#### Option B: Generate training data from raw audio
```bash
# Build PercepNet feature extractor
mkdir -p bin && cd bin
cmake .. -DTRAINING=1
make -j

# Generate training features (example)
cd ..
./bin/src/percepNet sampledata/speech/speech.pcm sampledata/noise/noise.pcm 4000 training.output

# Convert to HDF5 format
python3 utils/bin2h5.py training.output training.h5
```

#### Option C: Generate file lists for raw binary files
```bash
# Create file lists for raw .f32 files
find /path/to/training/data -name "*.f32" > train_filelist.txt
find /path/to/dev/data -name "*.f32" > dev_filelist.txt
```

## Running Enhanced Training

### 1. Basic Training Command
```bash
python rnn_train_with_diagnostics.py \
    --config config_enhanced.yaml \
    --train_filelist_path train_filelist.txt \
    --dev_filelist_path dev_filelist.txt \
    --out_dir ./experiments/run1
```

### 2. Training with Custom Parameters
```bash
python rnn_train_with_diagnostics.py \
    --config config_enhanced.yaml \
    --train_filelist_path train_filelist.txt \
    --dev_filelist_path dev_filelist.txt \
    --out_dir ./experiments/run1 \
    --loss_gamma 0.5 \
    --loss_C4 10.0 \
    --loss_eps 1e-8 \
    --clamp_predictions \
    --wandb_project "my-percepnet-project" \
    --wandb_run_name "experiment-1"
```

### 3. Resume from Checkpoint
```bash
python rnn_train_with_diagnostics.py \
    --config config_enhanced.yaml \
    --train_filelist_path train_filelist.txt \
    --dev_filelist_path dev_filelist.txt \
    --out_dir ./experiments/run1 \
    --pretrain ./experiments/run1/checkpoint-10000steps.pt
```

## Monitoring Training

### W&B Dashboard
- Go to https://wandb.ai
- Navigate to your project (default: "percepnet-enhanced")
- Monitor real-time metrics:
  - Loss components (Lg, Lq, Lr)
  - Gradient health metrics
  - Training visualizations
  - Ablation study results

### Local Monitoring
```bash
# View TensorBoard logs
tensorboard --logdir ./experiments/run1

# Check training logs
tail -f training.log

# Monitor GPU usage
nvidia-smi -l 1
```

## Understanding the Output

### Key Metrics to Watch

1. **Loss Components**:
   - `train/loss_Lg`: Gain prediction error
   - `train/loss_Lq`: Quartic gain penalty
   - `train/loss_Lr`: Residual prediction error
   - `train/total_loss`: Combined loss

2. **Gradient Health**:
   - `train/grad_norm`: Global gradient norm
   - `gradient_health_alerts`: Count of large gradients
   - `nan_count`: Count of NaN losses encountered

3. **Diagnostic Metrics**:
   - `diagnostics/grad_max`: Maximum gradient element
   - `diagnostics/prop_large_gt_1`: Proportion of gradients > 1
   - `diagnostics/outputs_min/max`: Model output range

### Expected Performance

**Normal Training Signs**:
- Loss decreasing steadily
- Gradient norm < 10.0 most of the time
- No NaN losses
- Output range stays in [0, 1]

**Warning Signs**:
- Frequent NaN losses (> 5% of steps)
- Gradient norm consistently > 100
- Loss not decreasing after 5000 steps
- Outputs going to extremes (0 or 1 constantly)

## Troubleshooting

### Common Issues

1. **W&B Login Issues**
```bash
# Re-login to W&B
wandb login --relogin
```

2. **CUDA Out of Memory**
```bash
# Reduce batch size in config_enhanced.yaml
batch_size: 5  # instead of 10
```

3. **Data Loading Errors**
```bash
# Check file paths exist
head -5 train_filelist.txt
ls -la $(head -1 train_filelist.txt)
```

4. **NaN Losses**
- The enhanced loss function should prevent most NaN issues
- If still occurring, try:
  - Increase `loss_eps` to 1e-6
  - Enable `clamp_predictions: true`
  - Reduce learning rate to 5e-5

### Performance Optimization

1. **Speed up training**:
```yaml
# In config_enhanced.yaml
num_workers: 8  # Increase data loading workers
batch_size: 20  # Increase if GPU memory allows
```

2. **Reduce memory usage**:
```yaml
# In config_enhanced.yaml
diagnostic_interval_steps: 5000  # Less frequent diagnostics
ablation_interval_steps: 20000   # Less frequent ablations
```

## File Structure After Training

```
experiments/run1/
├── config.yml                    # Saved configuration
├── checkpoint-5000steps.pt       # Periodic checkpoints
├── checkpoint-10000steps.pt
├── final-checkpoint-50000steps.pt  # Final model
├── diagnostics-5000steps.pt      # Diagnostic history
├── events.out.tfevents.*         # TensorBoard logs
└── training.log                  # Text logs
```

## Converting to C++ Inference

After training, convert to C++ format:

```bash
# Export trained model to C++ header
python3 dump_percepnet.py experiments/run1/final-checkpoint-50000steps.pt

# Build inference version
cd bin
cmake .. -DTRAINING=0
make -j

# Test inference
./src/percepNet_run input.pcm output.pcm
```

## Advanced Usage

### Custom Loss Parameters
Experiment with different loss configurations:

```yaml
# Aggressive compression (more perceptual)
loss_gamma: 0.3
loss_C4: 50.0

# Conservative approach (more MSE-like)  
loss_gamma: 0.7
loss_C4: 1.0

# Maximum stability
loss_eps: 1e-6
clamp_predictions: true
```

### Hyperparameter Search
Enable automatic ablation studies:

```yaml
run_initial_ablation: true
enable_periodic_ablation: true
ablation_gammas: [0.2, 0.3, 0.5, 0.7]
ablation_C4s: [0.1, 1.0, 10.0, 100.0]
```

## Support

If you encounter issues:

1. Check the training logs: `tail -100 training.log`
2. Verify your W&B dashboard shows metrics
3. Compare your setup with the working examples above
4. Review the diagnostic output for gradient health warnings

The enhanced training script provides comprehensive monitoring to help identify and resolve issues quickly.