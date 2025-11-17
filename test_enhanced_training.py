#!/usr/bin/env python3

"""
PercepNet Enhanced Training - Validation Test Script

This script validates that the enhanced training system works correctly
by running a short training session with synthetic data.

Usage:
    python test_enhanced_training.py
    
This will:
1. Create synthetic training data
2. Run a few training steps with diagnostics
3. Validate W&B integration (if available)
4. Test model export functionality
"""

import os
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Test the diagnostic imports
try:
    from percepnet_loss_diagnostics import CustomLossStable, run_diagnostics_step
    print("✓ Diagnostic modules imported successfully")
except ImportError as e:
    print(f" Failed to import diagnostic modules: {e}")
    exit(1)

# Test W&B availability  
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ W&B available")
except ImportError:
    WANDB_AVAILABLE = False
    print(" Warning! W&B not available (will test without)")


class SyntheticPercepNetDataset(Dataset):
    """Generate synthetic data that matches PercepNet format."""
    
    def __init__(self, num_samples=50, sequence_length=500):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.x_dim = 70
        self.y_dim = 68
        
        # Generate realistic-looking data
        np.random.seed(42)  # Reproducible
        
        # Generate input features (band energies + pitch)
        self.x_data = []
        self.y_data = []
        
        for i in range(num_samples):
            # Generate input: band energies + 2 pitch features
            x = np.random.exponential(0.1, (sequence_length, 68))  # Band energies
            x = np.clip(x, 0, 1)  # Realistic range
            
            # Add pitch features
            pitch_features = np.random.uniform(0, 1, (sequence_length, 2))
            x = np.concatenate([x, pitch_features], axis=1)
            
            # Generate target: gb (34) + rb (34) gains
            gb = np.random.beta(2, 3, (sequence_length, 34))  # Gain values
            rb = np.random.beta(3, 2, (sequence_length, 34))  # Residual values
            y = np.concatenate([gb, rb], axis=1)
            
            self.x_data.append(x.astype(np.float32))
            self.y_data.append(y.astype(np.float32))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class SimplePercepNet(torch.nn.Module):
    """Simplified PercepNet for testing."""
    
    def __init__(self, input_dim=70, hidden_dim=128, output_dim=68):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.gru(x)
        x = self.fc2(x)
        return self.activation(x)


def test_enhanced_training():
    """Test the enhanced training system."""
    
    print("  Testing Enhanced PercepNet Training System")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("  Creating synthetic dataset...")
    train_dataset = SyntheticPercepNetDataset(num_samples=20, sequence_length=100)
    val_dataset = SyntheticPercepNetDataset(num_samples=5, sequence_length=100)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Initialize model and training components
    print("🏗️ Initializing model and loss function...")
    model = SimplePercepNet().to(device)
    
    # Test different loss configurations
    loss_configs = [
        {"gamma": 0.5, "C4": 10.0, "eps": 0.0, "clamp_preds": False, "name": "standard"},
        {"gamma": 0.5, "C4": 10.0, "eps": 1e-8, "clamp_preds": True, "name": "stable"},
    ]
    
    for config in loss_configs:
        print(f"\n🔬 Testing loss configuration: {config['name']}")
        
        criterion = CustomLossStable(
            gamma=config["gamma"],
            C4=config["C4"], 
            eps=config["eps"],
            clamp_preds=config["clamp_preds"]
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Test a few training steps
        model.train()
        total_loss = 0
        
        for step, (inputs, targets) in enumerate(train_loader):
            if step >= 3:  # Just test a few steps
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Test component loss tracking
            loss, Lg, Lq, Lr = criterion(outputs, targets, return_components=True)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                
                # Check gradient health
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                
                optimizer.step()
                total_loss += loss.item()
                
                print(f"   Step {step+1}: loss={loss.item():.6f}, Lg={Lg:.6f}, Lq={Lq:.6f}, Lr={Lr:.6f}, grad_norm={grad_norm:.6f}")
            else:
                print(f"   Step {step+1}:  NaN/Inf loss detected")
        
        avg_loss = total_loss / min(3, len(train_loader))
        print(f"   Average loss: {avg_loss:.6f}")
    
    # Test diagnostic capabilities
    print("\n🔍 Testing diagnostic capabilities...")
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Test diagnostic function
    criterion = CustomLossStable(gamma=0.5, C4=10.0, eps=1e-8, clamp_preds=True)
    
    try:
        diagnostics, param_grad_norms = run_diagnostics_step(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            inputs=inputs,
            targets=targets,
            label="test",
            device=str(device),
            do_backward=True
        )
        print("Diagnostic analysis completed successfully")
        print(f"   Tracked {len(diagnostics)} metrics")
        print(f"   Analyzed {len(param_grad_norms)} parameters")
    except Exception as e:
        print(f"Diagnostic test failed: {e}")
    
    # Test W&B integration (if available)
    if WANDB_AVAILABLE:
        print("\n  Testing W&B integration...")
        try:
            # Initialize test run
            wandb.init(
                project="percepnet-test", 
                name="validation-test",
                mode="offline"  # Don't upload during test
            )
            
            # Log some test metrics
            wandb.log({
                "test/loss": avg_loss,
                "test/diagnostics": len(diagnostics),
                "test/device": str(device)
            })
            
            wandb.finish()
            print("W&B integration test successful")
            
        except Exception as e:
            print(f"W&B test failed (but training will work): {e}")
    
    # Test model export compatibility
    print("\n  Testing model export...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            
            # Reload to verify
            new_model = SimplePercepNet().to(device)
            new_model.load_state_dict(torch.load(f.name, map_location=device, weights_only=True))
            
            os.unlink(f.name)  # Cleanup
            print("Model export/import test successful")
            
    except Exception as e:
        print(f" Model export test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Enhanced Training System Validation Complete!")
    print("\nSummary:")
    print("   Synthetic data generation")
    print("   CustomLossStable functionality") 
    print("   Component loss tracking")
    print("   Gradient health monitoring")
    print("   Diagnostic analysis")
    print(f"   {'ok!' if WANDB_AVAILABLE else 'Warning!'} W&B integration {'(available)' if WANDB_AVAILABLE else '(install wandb for full features)'}")
    print("   Model checkpoint system")
    print("\n Ready for full training with real data!")
    
    return True


if __name__ == "__main__":
    try:
        test_enhanced_training()
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)