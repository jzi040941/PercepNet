# PercepNet Enhanced Training - Student Checklist

## ✅ Pre-Flight Checklist

Your supervisor has provided you with a complete enhanced training system. Here's what you need to verify before starting:

### 📁 Required Files (should all be present)
- [ ] `rnn_train_with_diagnostics.py` - Main training script  
- [ ] `percepnet_loss_diagnostics.py` - Enhanced loss & diagnostics
- [ ] `config_enhanced.yaml` - Configuration file
- [ ] `test_enhanced_training.py` - Validation test script
- [ ] `setup_enhanced_training.sh` - Automated setup script
- [ ] `TRAINING_GUIDE.md` - Complete documentation
- [ ] `requirements_enhanced.txt` - Python dependencies

### 🛠️ Setup Steps

1. **Run the automated setup**:
   ```bash
   ./setup_enhanced_training.sh
   ```

2. **Install and configure W&B** (recommended):
   ```bash
   pip install wandb
   wandb login
   ```
   Get your API key from https://wandb.ai/settings

3. **Test the installation**:
   ```bash
   python3 test_enhanced_training.py
   ```
   ✅ Should show: "Enhanced Training System Validation Complete!"

### 📊 Data Preparation

Choose your data format:

**Option A: Raw binary files (.f32)**
```bash
find /path/to/your/training/data -name "*.f32" > train_filelist.txt
find /path/to/your/dev/data -name "*.f32" > dev_filelist.txt
```

**Option B: Generate from audio** (if you have .pcm files)
```bash
# Build feature extractor
mkdir -p bin && cd bin
cmake .. -DTRAINING=1 
make -j
cd ..

# Generate features
./bin/src/percepNet speech.pcm noise.pcm 4000 training.output
python3 utils/bin2h5.py training.output training.h5
```

### 🚀 Start Training

**Basic command**:
```bash
python3 rnn_train_with_diagnostics.py \
    --config config_enhanced.yaml \
    --train_filelist_path train_filelist.txt \
    --dev_filelist_path dev_filelist.txt \
    --out_dir ./experiments/my_first_run
```

**With custom parameters**:
```bash
python3 rnn_train_with_diagnostics.py \
    --config config_enhanced.yaml \
    --train_filelist_path train_filelist.txt \
    --dev_filelist_path dev_filelist.txt \
    --out_dir ./experiments/stable_training \
    --loss_gamma 0.5 \
    --loss_C4 10.0 \
    --loss_eps 1e-8 \
    --clamp_predictions \
    --wandb_project "my-percepnet-experiments" \
    --wandb_run_name "stable-loss-experiment-1"
```

### 📊 Monitor Training

1. **W&B Dashboard**: https://wandb.ai (real-time metrics)
2. **Local logs**: `tail -f training.log`
3. **TensorBoard**: `tensorboard --logdir ./experiments`
4. **GPU usage**: `nvidia-smi -l 1`

### 🔍 What to Watch For

**✅ Good Signs:**
- Loss decreasing steadily
- No NaN losses
- Gradient norm < 10.0 most of the time
- Component losses (Lg, Lq, Lr) all contributing
- Model outputs staying in [0, 1] range

**⚠️ Warning Signs:**
- NaN losses (should be rare with enhanced loss)
- Gradient norm > 100 consistently
- Loss not decreasing after 5000 steps
- W&B showing gradient explosion warnings

### 🆘 Troubleshooting

**If you see NaN losses:**
```yaml
# In config_enhanced.yaml, increase stability:
loss_eps: 1e-6
clamp_predictions: true
learning_rate: 5e-5
```

**If GPU runs out of memory:**
```yaml
# In config_enhanced.yaml:
batch_size: 5  # instead of 10
num_workers: 2  # instead of 4
```

**If training is slow:**
```yaml
# In config_enhanced.yaml:
diagnostic_interval_steps: 5000  # less frequent
ablation_interval_steps: 20000   # less frequent
```

### 🎯 Expected Timeline

- **First 1000 steps**: Loss should start decreasing
- **5000 steps**: Clear downward trend, no gradient issues
- **10000 steps**: Loss components stabilizing
- **50000+ steps**: Good convergence for full dataset

### 📈 Success Metrics

After training, you should have:
- [ ] Steady loss decrease over time
- [ ] No NaN/Inf issues in training logs
- [ ] Model checkpoints saved regularly
- [ ] W&B dashboard showing component losses
- [ ] Diagnostic reports showing healthy gradients
- [ ] Final model ready for C++ export

### 🔄 Export to C++ (after training)

```bash
# Export trained model to C++ format
python3 dump_percepnet.py experiments/my_first_run/final-checkpoint-50000steps.pt

# Build inference version
cd bin
cmake .. -DTRAINING=0
make -j

# Test inference
./src/percepNet_run input.pcm output.pcm
```

## 📞 Support

If you encounter issues:
1. Check this checklist first
2. Review `TRAINING_GUIDE.md` for detailed help
3. Check your W&B dashboard for diagnostic alerts
4. Look at `training.log` for error messages
5. Verify your setup with `python3 test_enhanced_training.py`

The enhanced system provides comprehensive monitoring and should help you identify and resolve issues quickly!

---
**Note**: This enhanced training system includes significant improvements over the standard PercepNet training, including numerical stability, comprehensive diagnostics, and real-time monitoring. Make full use of these features to understand your training process!