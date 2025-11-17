#!/bin/bash

# PercepNet Enhanced Training - Quick Setup Script
# Run this script to set up everything your student needs

echo "🚀 PercepNet Enhanced Training Setup"
echo "====================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.7+"; exit 1; }

# Create virtual environment (recommended)
echo "🔧 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements_enhanced.txt" ]; then
    pip install -r requirements_enhanced.txt
    echo "✅ Installed enhanced requirements"
else
    echo "⚠ requirements_enhanced.txt not found, installing basic requirements..."
    pip install torch numpy h5py tqdm matplotlib pyyaml tensorboardx wandb Pillow torchvision
fi

# Login to W&B (interactive)
echo "🔑 Setting up Weights & Biases..."
if command -v wandb &> /dev/null; then
    echo "Run 'wandb login' to set up W&B (optional but recommended)"
    echo "You can get your API key from https://wandb.ai/settings"
else
    echo "⚠ wandb not installed, W&B features will be disabled"
fi

# Test the installation
echo "🧪 Testing installation..."
if python3 test_enhanced_training.py; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📚 Next steps for your student:"
    echo "1. Prepare training data (see TRAINING_GUIDE.md)"
    echo "2. Edit config_enhanced.yaml as needed"
    echo "3. Run training:"
    echo "   python3 rnn_train_with_diagnostics.py \\"
    echo "     --config config_enhanced.yaml \\"
    echo "     --train_filelist_path train_files.txt \\"
    echo "     --dev_filelist_path dev_files.txt \\"
    echo "     --out_dir ./experiments/run1"
    echo ""
    echo "📊 Monitor training at:"
    echo "   - W&B Dashboard: https://wandb.ai"
    echo "   - TensorBoard: tensorboard --logdir ./experiments"
    echo ""
    echo "📖 Full documentation in TRAINING_GUIDE.md"
else
    echo "❌ Installation test failed. Please check error messages above."
    exit 1
fi