# PercepNet Loss Diagnostics - CLI Usage Examples

This document provides practical command-line examples for using the PercepNet loss diagnostics system. These examples demonstrate how to run different diagnostic scenarios with copy-paste commands.

## **1. Quick Smoke Test - Verify Installation**

Test that the diagnostic module works with dummy data:

```bash
# Test the diagnostic module works with dummy data
cd /home/jhjerrild/builds/PercepNet
python3 percepnet_loss_diagnostics.py
```

**Expected output:**
```
[DIAG smoke-test] loss_total=1.234567e+00  Lg=4.567890e-01  Lq=2.345678e-02  Lr=7.543210e-01
[DIAG smoke-test] outputs_min=-1.234567e+00  outputs_max=2.345678e+00
[DIAG smoke-test] grad_max=5.678901e+00  prop>|g|>1=0.1234  prop>|g|>10=0.0000  total_grad_norm=1.234567e+01
```

## **2. Training with Original vs Stable Loss Comparison**

Compare training behavior between original and stable loss implementations:

```bash
# Train with original CustomLoss (baseline)
cd /home/jhjerrild/builds/PercepNet
python3 rnn_train.py --loss=original --epochs=10 --output=baseline_results

# Train with stable loss (your implementation) 
python3 rnn_train.py --loss=stable --eps=1e-8 --clamp=true --epochs=10 --output=stable_results

# Compare tensorboard logs
tensorboard --logdir=runs --port=6006
```

## **3. Single-Step Gradient Health Check**

Check gradient health of an existing model:

```bash
python3 -c "
from percepnet_loss_diagnostics import run_diagnostics_step, CustomLossStable
from rnn_train import PercepNet, h5Dataset
import torch

# Load model and data
model = PercepNet()
model.load_state_dict(torch.load('model.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
dataset = h5Dataset('training.h5')
inputs, targets = dataset[0]

# Run health check
loss_fn = CustomLossStable(gamma=0.5, C4=10.0, eps=1e-8, clamp_preds=True)
diagnostics, _ = run_diagnostics_step(model, optimizer, loss_fn, 
                                    inputs.unsqueeze(0), targets.unsqueeze(0),
                                    label='health_check', device='cpu')
"
```

## **4. Hyperparameter Ablation Study**

Run comprehensive ablation study across different loss configurations:

```bash
# Run comprehensive ablation study
python3 -c "
from percepnet_loss_diagnostics import run_ablation_suite, CustomLossStable
from rnn_train import PercepNet, h5Dataset
import torch, json

# Setup
model = PercepNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
dataset = h5Dataset('training.h5')
inputs, targets = dataset[0]

# Run ablation
results = run_ablation_suite(
    model, optimizer, inputs.unsqueeze(0), targets.unsqueeze(0),
    device='cpu',
    gammas=(1.0, 0.75, 0.5, 0.25),
    C4s=(0.0, 1.0, 5.0, 10.0, 20.0),
    eps_values=(0.0, 1e-10, 1e-8, 1e-6),
    clamp_options=(False, True)
)

# Save results
with open('ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to ablation_results.json')
"
```

**Analyze ablation results:**
```bash
python3 -c "
import json
with open('ablation_results.json') as f:
    results = json.load(f)

print('=== ABLATION STUDY RESULTS ===')
for config, metrics in sorted(results.items(), key=lambda x: x[1]['loss_total']):
    print(f'{config:25s}: loss={metrics[\"loss_total\"]:.6e} grad_max={metrics[\"grad_max\"]:.6e} stable={metrics[\"prop_large_gt_10\"]:.4f}')
"
```

## **5. Monitor Training with Periodic Health Checks**

Create enhanced training script with integrated diagnostics:

```bash
# Create enhanced training script with diagnostics
cat > train_with_diagnostics.py << 'EOF'
from rnn_train import PercepNet, h5Dataset
from percepnet_loss_diagnostics import CustomLossStable, run_diagnostics_step
import torch
from tensorboardX import SummaryWriter

# Setup
model = PercepNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = CustomLossStable(gamma=0.5, C4=10.0, eps=1e-8, clamp_preds=True)
dataset = h5Dataset("training.h5")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
writer = SummaryWriter()

# Training with diagnostics
for epoch in range(100):
    for i, (inputs, targets) in enumerate(train_loader):
        # Normal training step
        optimizer.zero_grad()
        outputs = model(inputs)
        total_loss, Lg, Lq, Lr = criterion(outputs, targets, return_components=True)
        total_loss.backward()
        optimizer.step()
        
        # Log components
        step = epoch * len(train_loader) + i
        writer.add_scalar('Loss/Total', total_loss.item(), step)
        writer.add_scalar('Loss/Lg_gain', Lg.item(), step)
        writer.add_scalar('Loss/Lq_quartic', Lq.item(), step)
        writer.add_scalar('Loss/Lr_residual', Lr.item(), step)
        
        # Periodic health check
        if i % 500 == 0:
            diagnostics, _ = run_diagnostics_step(
                model, optimizer, criterion, inputs, targets,
                label=f"epoch{epoch}_iter{i}", do_backward=False
            )
            
            # Alert on issues
            if diagnostics['grad_max'] > 100.0:
                print(f"⚠️  Large gradients detected at epoch {epoch}, iter {i}")
            if diagnostics['prop_large_gt_10'] > 0.01:
                print(f"⚠️  {diagnostics['prop_large_gt_10']:.2%} of gradients > 10")

print("Training complete with diagnostics!")
EOF

# Run enhanced training
python3 train_with_diagnostics.py

# Monitor in real-time
tensorboard --logdir=runs --reload_interval=10
```

## **6. Compare Loss Components Between Models**

Compare diagnostic metrics between two different trained models:

```bash
python3 -c "
from percepnet_loss_diagnostics import run_diagnostics_step, CustomLossStable
from rnn_train import PercepNet, h5Dataset
import torch

# Load models
model1 = PercepNet()
model1.load_state_dict(torch.load('model_baseline.pt'))

model2 = PercepNet()
model2.load_state_dict(torch.load('model_stable.pt'))

# Load test data
dataset = h5Dataset('training.h5')
inputs, targets = dataset[0]
optimizer = torch.optim.Adam(model1.parameters())
loss_fn = CustomLossStable()

print('=== MODEL COMPARISON ===')
for name, model in [('Baseline', model1), ('Stable', model2)]:
    diagnostics, _ = run_diagnostics_step(
        model, optimizer, loss_fn, inputs.unsqueeze(0), targets.unsqueeze(0),
        label=name, device='cpu'
    )
    print(f'{name:10s}: Total={diagnostics[\"loss_total\"]:.6e} Lg={diagnostics[\"Lg\"]:.6e} Lq={diagnostics[\"Lq\"]:.6e} Lr={diagnostics[\"Lr\"]:.6e}')
"
```

## **7. Export Comprehensive Diagnostic Report**

Generate a complete diagnostic report comparing multiple configurations:

```bash
python3 -c "
from percepnet_loss_diagnostics import run_diagnostics_step, CustomLossStable
from rnn_train import PercepNet, h5Dataset
import torch, json, datetime

# Setup
model = PercepNet()
model.load_state_dict(torch.load('model.pt'))
dataset = h5Dataset('training.h5')
optimizer = torch.optim.Adam(model.parameters())

# Test multiple configurations
configs = [
    ('baseline', {'gamma': 0.5, 'C4': 10.0, 'eps': 0.0, 'clamp_preds': False}),
    ('stable_minimal', {'gamma': 0.5, 'C4': 10.0, 'eps': 1e-10, 'clamp_preds': False}),
    ('stable_full', {'gamma': 0.5, 'C4': 10.0, 'eps': 1e-8, 'clamp_preds': True}),
]

report = {
    'timestamp': datetime.datetime.now().isoformat(),
    'model_path': 'model.pt',
    'configurations': {}
}

for name, config in configs:
    loss_fn = CustomLossStable(**config)
    inputs, targets = dataset[0]
    diagnostics, param_norms = run_diagnostics_step(
        model, optimizer, loss_fn, inputs.unsqueeze(0), targets.unsqueeze(0),
        label=name, device='cpu'
    )
    
    report['configurations'][name] = {
        'config': config,
        'diagnostics': diagnostics,
        'top_param_norms': param_norms[:5]  # Top 5 parameters
    }

# Save report
with open(f'diagnostic_report_{datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print('Diagnostic report saved!')
"
```

## **8. Quick Gradient Health Alerts**

Set up automated gradient health monitoring with alerts:

```bash
python3 -c "
from percepnet_loss_diagnostics import _collect_grad_stats
from rnn_train import PercepNet
import torch

model = PercepNet()
model.load_state_dict(torch.load('model.pt'))

# Simulate some gradients (in real use, this would be after loss.backward())
for param in model.parameters():
    param.grad = torch.randn_like(param) * 0.01  # Mock gradients

# Check gradient health
grad_max, total_grad_norm, params_with_grad, count_gt_1, count_gt_10, total_grad_count, top_param_norms = _collect_grad_stats(model)

# Health alerts
print('=== GRADIENT HEALTH CHECK ===')
if grad_max > 50.0:
    print(f'🚨 CRITICAL: Maximum gradient {grad_max:.2e} exceeds threshold!')
elif grad_max > 10.0:
    print(f'⚠️  WARNING: Large gradient detected: {grad_max:.2e}')
else:
    print(f'✅ Gradients healthy: max={grad_max:.2e}')

if count_gt_10 / total_grad_count > 0.005:  # >0.5% of gradients are large
    print(f'🚨 CRITICAL: {100*count_gt_10/total_grad_count:.1f}% of gradients > 10')
elif count_gt_1 / total_grad_count > 0.1:   # >10% of gradients are moderate
    print(f'⚠️  WARNING: {100*count_gt_1/total_grad_count:.1f}% of gradients > 1')
else:
    print('✅ Gradient distribution healthy')

print(f'Total gradient norm: {total_grad_norm:.6e}')
"
```

## **9. Component-wise Loss Analysis**

Analyze which loss components dominate during training:

```bash
python3 -c "
from percepnet_loss_diagnostics import CustomLossStable
from rnn_train import PercepNet, h5Dataset
import torch

# Setup
model = PercepNet()
model.load_state_dict(torch.load('model.pt'))
dataset = h5Dataset('training.h5')
criterion = CustomLossStable(gamma=0.5, C4=10.0, eps=1e-8, clamp_preds=True)

# Analyze multiple samples
print('=== LOSS COMPONENT ANALYSIS ===')
print('Sample    Total_Loss    Lg_gain      Lq_quartic   Lr_residual  Dominant_Component')
print('-' * 80)

for i in range(10):  # Analyze first 10 samples
    inputs, targets = dataset[i]
    outputs = model(inputs.unsqueeze(0))
    
    total_loss, Lg, Lq, Lr = criterion(outputs, targets.unsqueeze(0), return_components=True)
    
    # Find dominant component
    components = {'Lg': Lg.item(), 'Lq': Lq.item() * 10.0, 'Lr': Lr.item()}  # Scale Lq by C4=10
    dominant = max(components, key=components.get)
    
    print(f'{i:6d}    {total_loss.item():.6e}  {Lg.item():.6e}  {Lq.item():.6e}  {Lr.item():.6e}  {dominant}')
"
```

## **10. Live Training Monitoring Dashboard**

Set up a simple live monitoring script:

```bash
# Create live monitoring script
cat > monitor_training.py << 'EOF'
import time, json, os
from datetime import datetime

def monitor_training_logs(log_file='training_diagnostics.log'):
    print("🔍 Monitoring PercepNet training diagnostics...")
    print("Press Ctrl+C to stop\n")
    
    if not os.path.exists(log_file):
        print(f"❌ Log file {log_file} not found. Make sure training is running with diagnostics enabled.")
        return
    
    last_size = 0
    while True:
        try:
            current_size = os.path.getsize(log_file)
            if current_size > last_size:
                with open(log_file, 'r') as f:
                    f.seek(last_size)
                    new_lines = f.read()
                    
                for line in new_lines.strip().split('\n'):
                    if line.startswith('[DIAG'):
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {line}")
                        
                        # Parse and alert on issues
                        if 'grad_max=' in line:
                            grad_max_str = line.split('grad_max=')[1].split()[0]
                            grad_max = float(grad_max_str)
                            if grad_max > 50.0:
                                print(f"  🚨 HIGH GRADIENTS DETECTED: {grad_max:.2e}")
                
                last_size = current_size
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training_logs()
EOF

# Run monitoring (in separate terminal while training)
python3 monitor_training.py
```

## **Usage Tips**

### **Getting Started**
1. Start with **Quick Smoke Test** (#1) to verify installation
2. Run **Single-Step Gradient Health Check** (#3) on your existing model
3. Try **Component-wise Loss Analysis** (#9) to understand your current loss behavior

### **For Training**
- Use **Training with Periodic Health Checks** (#5) for new training runs
- Use **Compare Loss Components** (#6) to evaluate different models
- Use **Hyperparameter Ablation Study** (#4) to optimize loss parameters

### **For Debugging**
- Use **Gradient Health Alerts** (#8) when experiencing training instabilities
- Use **Export Diagnostic Report** (#7) for comprehensive analysis
- Use **Live Training Monitoring** (#10) for real-time oversight

### **Integration with Existing Workflow**
Replace your current loss function in `rnn_train.py` line 199:
```python
# From:
criterion = CustomLoss()

# To:
from percepnet_loss_diagnostics import CustomLossStable
criterion = CustomLossStable(gamma=0.5, C4=10.0, eps=1e-8, clamp_preds=True)
```

These examples provide practical, copy-paste commands for different diagnostic scenarios, making it easy to demonstrate and use the enhanced loss diagnostics system.