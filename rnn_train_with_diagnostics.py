#!/usr/bin/env python3

"""
PercepNet Training Script with Enhanced Diagnostics and W&B Logging

This training script provides:
- Complete integration with Weights & Biases (wandb) for experiment tracking
- Enhanced loss function with numerical stability (CustomLossStable)
- Real-time gradient health monitoring and component analysis
- Comprehensive diagnostic logging with configurable intervals
- Ablation study capabilities for hyperparameter exploration
- Backward compatibility with existing PercepNet architecture

Usage:
    python rnn_train_with_diagnostics.py --config config.yaml --train_filelist_path train.txt --dev_filelist_path dev.txt --out_dir ./experiments

Requirements:
    pip install wandb torch numpy h5py tqdm matplotlib pyyaml tensorboardx

Created by: PercepNet Team
Enhanced with: Advanced diagnostics and W&B integration
"""

import argparse
import logging
import os
import io
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import yaml
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import PIL.Image
from torchvision.transforms import ToTensor

# W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("WARNING: wandb not available. Install with: pip install wandb")
    print("         Falling back to tensorboard logging only.")
    WANDB_AVAILABLE = False

# TensorBoard fallback
from tensorboardX import SummaryWriter

# Import our enhanced diagnostics
from percepnet_loss_diagnostics import CustomLossStable, run_diagnostics_step, run_ablation_suite

# Matplotlib backend for server environments
plt.switch_backend('agg')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class CppRawListDataset(Dataset):
    """Dataset for loading CPP-generated raw features from file lists."""
    
    def __init__(self, filelist_path, train_length_size=500):
        self.train_length_size = train_length_size
        self.filelist_path = filelist_path
        self.x_dim = 70
        self.y_dim = 68

        with open(filelist_path, "r") as f:
            self.filelist = [filepath.rstrip('\n') for filepath in f.readlines()]
        self.nb_sequences = len(self.filelist)

        logging.info(f"Loaded {self.nb_sequences} sequences from {filelist_path}")

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        try:
            with open(self.filelist[index], 'rb') as cpp_out:
                all_data = np.fromfile(cpp_out, np.float32)
                all_data = np.reshape(all_data, (self.train_length_size, 138))
                # Band energy scaling (30x compensation for low energy)
                all_data[:, :68] = all_data[:, :68] * 30

            x = all_data[:, :self.x_dim]
            y = all_data[:, self.x_dim:]
            return (x, y)
        except Exception as e:
            logging.error(f"Error loading file {self.filelist[index]}: {e}")
            raise


class h5DirDataset(Dataset):
    """Dataset for loading HDF5 files from a directory."""
    
    def __init__(self, h5_dir_path, train_length_size=500):
        self.train_length_size = train_length_size
        self.h5_dir_path = h5_dir_path
        self.x_dim = 70
        self.y_dim = 68

        self.h5_filelist = glob.glob(os.path.join(h5_dir_path, "*.h5"))
        self.nb_sequences = len(self.h5_filelist)

        logging.info(f"Loaded {self.nb_sequences} h5 files from {h5_dir_path}")

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        with h5py.File(self.h5_filelist[index], 'r') as hf:
            all_data = hf['data'][:]
        x = all_data[:, :self.x_dim]
        y = all_data[:, self.x_dim:]
        return (x, y)


class PercepNet(nn.Module):
    """PercepNet architecture as defined in the original paper."""
    
    def __init__(self, input_dim=70):
        super(PercepNet, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv1d(128, 512, 5, stride=1, padding=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(512, 512, 3, stride=1, padding=2), nn.Tanh())
        
        self.gru1 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru2 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru3 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_gb = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_rb = nn.GRU(1024, 128, 1, batch_first=True)
        
        self.fc_gb = nn.Sequential(nn.Linear(512*5, 34), nn.Sigmoid())
        self.fc_rb = nn.Sequential(nn.Linear(128, 34), nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x)
        x = x.permute([0, 2, 1])  # B, D, T
        x = self.conv1(x)
        x = x[:, :, :-4]
        convout = self.conv2(x)
        convout = convout[:, :, :-2]  # align with C++ DNN
        convout = convout.permute([0, 2, 1])  # B, T, D
       
        gru1_out, gru1_state = self.gru1(convout)
        gru2_out, gru2_state = self.gru2(gru1_out)
        gru3_out, gru3_state = self.gru3(gru2_out)
        gru_gb_out, gru_gb_state = self.gru_gb(gru3_out)
        
        concat_gb_layer = torch.cat((convout, gru1_out, gru2_out, gru3_out, gru_gb_out), -1)
        gb = self.fc_gb(concat_gb_layer)

        concat_rb_layer = torch.cat((gru3_out, convout), -1)
        rnn_rb_out, gru_rb_state = self.gru_rb(concat_rb_layer)
        rb = self.fc_rb(rnn_rb_out)

        output = torch.cat((gb, rb), -1)
        return output


def gen_plot(y, y_hat):
    """Generate comparison plot for tensorboard/wandb visualization."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(y_hat.T, interpolation='none', cmap=plt.cm.jet, origin='lower', aspect='auto')
    plt.title('Predicted')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(y.T, interpolation='none', cmap=plt.cm.jet, origin='lower', aspect='auto')
    plt.title('Target')
    plt.colorbar()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Prevent memory leak
    return buf


class EnhancedTrainer(object):
    """Enhanced trainer with diagnostic capabilities and W&B integration."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        args,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize enhanced trainer."""
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.args = args
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # Initialize logging systems
        self.writer = SummaryWriter(config["out_dir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        
        # Diagnostic tracking
        self.diagnostic_history = []
        self.gradient_health_alerts = 0
        self.nan_count = 0
        
        # Initialize W&B if available
        self.use_wandb = WANDB_AVAILABLE and config.get("use_wandb", True)
        if self.use_wandb:
            self._init_wandb()
        
        logging.info(f"Enhanced trainer initialized - W&B: {self.use_wandb}")

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb_config = {
            "learning_rate": self.config.get("learning_rate", 0.0001),
            "batch_size": self.config["batch_size"],
            "train_max_steps": self.config["train_max_steps"],
            "gamma": getattr(self.criterion, 'gamma', 0.5),
            "C4": getattr(self.criterion, 'C4', 10.0),
            "eps": getattr(self.criterion, 'eps', 0.0),
            "clamp_preds": getattr(self.criterion, 'clamp_preds', False),
            "architecture": "PercepNet",
            "dataset_type": type(self.data_loader["train"].dataset).__name__,
        }
        wandb_config.update(self.config)
        
        run_name = self.config.get("wandb_run_name", f"percepnet_{int(time.time())}")
        
        wandb.init(
            project=self.config.get("wandb_project", "percepnet-enhanced"),
            name=run_name,
            config=wandb_config,
            resume=self.config.get("wandb_resume", "allow"),
            notes=self.config.get("wandb_notes", "Enhanced training with diagnostics"),
        )
        wandb.watch(self.model, log="all", log_freq=self.config.get("wandb_watch_freq", 1000))

    def run(self):
        """Run enhanced training with diagnostics."""
        logging.info("Starting enhanced training with diagnostics...")
        
        self.tqdm = tqdm(
            initial=self.steps, 
            total=self.config["train_max_steps"], 
            desc="[Enhanced Training]"
        )
        
        # Run initial ablation study if requested
        if self.config.get("run_initial_ablation", False):
            self._run_initial_ablation()
        
        while not self.finish_train:
            try:
                self._train_epoch()
            except Exception as e:
                logging.error(f"Training error: {e}")
                if self.use_wandb:
                    wandb.log({"error": str(e)})
                raise

        self.tqdm.close()
        logging.info("Finished enhanced training.")
        
        if self.use_wandb:
            wandb.finish()

    def _train_step(self, batch):
        """Enhanced train step with diagnostics."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        # Enhanced loss computation with component tracking
        if hasattr(self.criterion, 'forward') and 'return_components' in self.criterion.forward.__code__.co_varnames:
            loss, Lg, Lq, Lr = self.criterion(outputs, targets, return_components=True)
            
            # Log component losses
            self.total_train_loss["train/loss_Lg"] += float(Lg)
            self.total_train_loss["train/loss_Lq"] += float(Lq)
            self.total_train_loss["train/loss_Lr"] += float(Lr)
        else:
            loss = self.criterion(outputs, targets)
        
        # Check for NaN before backward
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            logging.warning(f"NaN/Inf loss detected at step {self.steps} (count: {self.nan_count})")
            if self.use_wandb:
                wandb.log({"nan_count": self.nan_count, "step": self.steps})
            return  # Skip this step
        
        loss.backward()
        
        # Gradient health check
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
        if grad_norm > self.config.get("gradient_clip_threshold", 10.0):
            self.gradient_health_alerts += 1
            if self.config.get("clip_gradients", True):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config.get("gradient_clip_value", 1.0))
        
        self.optimizer.step()
        
        # Update tracking
        self.total_train_loss["train/total_loss"] += loss.item()
        self.total_train_loss["train/grad_norm"] += grad_norm.item()
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Enhanced train epoch with periodic diagnostics."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            self._train_step(batch)
            
            # Enhanced logging intervals
            if self.steps % self.config.get("log_interval_steps", 100) == 0:
                self._check_log_interval()
            
            if self.steps % self.config.get("eval_interval_steps", 1000) == 0:
                self._check_eval_interval()
            
            if self.steps % self.config.get("save_interval_steps", 5000) == 0:
                self._check_save_interval()
                
            # Diagnostic interval (detailed analysis)
            if self.steps % self.config.get("diagnostic_interval_steps", 2000) == 0:
                self._run_diagnostic_step()
            
            # Ablation study interval (hyperparameter exploration)
            if self.steps > 0 and self.steps % self.config.get("ablation_interval_steps", 10000) == 0:
                self._run_periodic_ablation()
            
            if self.finish_train:
                return

        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Completed epoch {self.epochs} ({train_steps_per_epoch} steps)")

    def _run_diagnostic_step(self):
        """Run comprehensive diagnostic analysis."""
        logging.info(f"Running diagnostic analysis at step {self.steps}...")
        
        # Get a sample batch for diagnostics
        try:
            sample_batch = next(iter(self.data_loader["train"]))
            inputs, targets = sample_batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Run diagnostics with current criterion
            diagnostics, param_grad_norms = run_diagnostics_step(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.criterion,
                inputs=inputs,
                targets=targets,
                label=f"step_{self.steps}",
                device=str(self.device),
                do_backward=True,
                keep_graph=False
            )
            
            # Store diagnostic history
            diagnostics["step"] = self.steps
            diagnostics["epoch"] = self.epochs
            self.diagnostic_history.append(diagnostics)
            
            # Enhanced W&B logging
            if self.use_wandb:
                wandb.log({
                    f"diagnostics/{k}": v for k, v in diagnostics.items() 
                    if isinstance(v, (int, float))
                }, step=self.steps)
                
                # Log top parameter gradient norms
                for i, (name, norm) in enumerate(param_grad_norms[:5]):
                    wandb.log({f"param_grad_norms/{name}": norm}, step=self.steps)
            
            # TensorBoard logging
            for k, v in diagnostics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"diagnostics/{k}", v, self.steps)
                    
        except Exception as e:
            logging.warning(f"Diagnostic step failed: {e}")

    def _run_initial_ablation(self):
        """Run initial ablation study to validate hyperparameters."""
        logging.info("Running initial ablation study...")
        
        try:
            sample_batch = next(iter(self.data_loader["train"]))
            inputs, targets = sample_batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Define ablation parameters
            gammas = self.config.get("ablation_gammas", [0.3, 0.5, 0.7])
            C4s = self.config.get("ablation_C4s", [1.0, 10.0, 50.0])
            eps_values = self.config.get("ablation_eps", [0.0, 1e-8, 1e-6])
            
            results = run_ablation_suite(
                model=self.model,
                optimizer=self.optimizer,
                inputs=inputs,
                targets=targets,
                device=str(self.device),
                gammas=gammas,
                C4s=C4s,
                eps_values=eps_values,
                clamp_options=[False, True]
            )
            
            # Log ablation results
            if self.use_wandb:
                for variant, metrics in results.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            wandb.log({f"ablation_initial/{variant}/{metric}": value})
                            
        except Exception as e:
            logging.warning(f"Initial ablation failed: {e}")

    def _run_periodic_ablation(self):
        """Run periodic ablation for hyperparameter monitoring."""
        if not self.config.get("enable_periodic_ablation", False):
            return
            
        logging.info(f"Running periodic ablation at step {self.steps}...")
        # Similar to initial ablation but with reduced parameter space
        try:
            sample_batch = next(iter(self.data_loader["train"]))
            inputs, targets = sample_batch
            
            # Quick ablation with fewer variants
            results = run_ablation_suite(
                model=self.model,
                optimizer=self.optimizer,
                inputs=inputs.to(self.device),
                targets=targets.to(self.device),
                device=str(self.device),
                gammas=[0.5],  # Current gamma only
                C4s=[1.0, self.config.get("C4", 10.0), 50.0],
                eps_values=[0.0, 1e-8],
                clamp_options=[False, True]
            )
            
            if self.use_wandb:
                for variant, metrics in results.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            wandb.log({f"ablation_periodic/{variant}/{metric}": value}, step=self.steps)
                            
        except Exception as e:
            logging.warning(f"Periodic ablation failed: {e}")

    @torch.no_grad()
    def _eval_step(self, batch):
        """Enhanced evaluation step."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        outputs = self.model(inputs)
        
        # Enhanced loss computation
        if hasattr(self.criterion, 'forward') and 'return_components' in self.criterion.forward.__code__.co_varnames:
            loss, Lg, Lq, Lr = self.criterion(outputs, targets, return_components=True)
            self.total_eval_loss["eval/loss_Lg"] += float(Lg)
            self.total_eval_loss["eval/loss_Lq"] += float(Lq)
            self.total_eval_loss["eval/loss_Lr"] += float(Lr)
        else:
            loss = self.criterion(outputs, targets)
            
        self.total_eval_loss["eval/total_loss"] += loss.item()

    def _eval_epoch(self):
        """Enhanced evaluation epoch."""
        logging.info(f"(Steps: {self.steps}) Starting evaluation...")
        self.model.eval()

        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            self._eval_step(batch)
            
            # Save intermediate result from first batch
            if eval_steps_per_epoch == 1:
                self._generate_and_save_intermediate_result(batch)

        # Average and log evaluation losses
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}")

        self._write_to_tensorboard(self.total_eval_loss)
        
        if self.use_wandb:
            wandb.log(self.total_eval_loss, step=self.steps)

        # Reset and restore
        self.total_eval_loss = defaultdict(float)
        self.model.train()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate visualization results."""
        x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model(x_batch)

        # Generate visualization for first sample
        y, y_ = y_batch[0].cpu().numpy(), y_batch_[0].cpu().numpy()
        plot_buf = gen_plot(y, y_)
        
        # TensorBoard logging
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image)
        self.writer.add_image('predictions/rb_gb_comparison', image, self.steps)
        
        # W&B logging
        if self.use_wandb:
            wandb.log({
                "predictions/comparison": wandb.Image(plot_buf, caption=f"Step {self.steps}")
            }, step=self.steps)

    def _write_to_tensorboard(self, loss_dict):
        """Write metrics to tensorboard."""
        for key, value in loss_dict.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config["out_dir"], f"checkpoint-{self.steps}steps.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint @ {self.steps} steps")
        
        # Save diagnostic history
        if self.diagnostic_history:
            diag_path = os.path.join(self.config["out_dir"], f"diagnostics-{self.steps}steps.pt")
            torch.save(self.diagnostic_history, diag_path)

    def _check_eval_interval(self):
        """Run evaluation."""
        self._eval_epoch()

    def _check_log_interval(self):
        """Log training metrics."""
        log_interval = self.config.get("log_interval_steps", 100)
        
        for key in self.total_train_loss.keys():
            self.total_train_loss[key] /= log_interval
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}")
        
        self._write_to_tensorboard(self.total_train_loss)
        
        if self.use_wandb:
            wandb_log = dict(self.total_train_loss)
            wandb_log.update({
                "gradient_health_alerts": self.gradient_health_alerts,
                "nan_count": self.nan_count,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            wandb.log(wandb_log, step=self.steps)

        # Reset counters
        self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        """Check if training should finish."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


def main():
    """Main training function with enhanced diagnostics."""
    parser = argparse.ArgumentParser(
        description="Train PercepNet with Enhanced Diagnostics and W&B Logging"
    )
    
    # Training parameters
    parser.add_argument("--train_length_size", default=2000, type=int,
                       help="RNN network train length size")
    parser.add_argument("--train_max_steps", default=100000, type=int,
                       help="Maximum training steps")
    parser.add_argument("--train_filelist_path", type=str, required=True,
                       help="Training filelist path")
    parser.add_argument("--dev_filelist_path", type=str, required=True,
                       help="Development filelist path")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--config", type=str, required=True,
                       help="YAML configuration file")
    parser.add_argument("--pretrain", default="", type=str,
                       help="Pretrained checkpoint path")
    
    # Enhanced diagnostic parameters
    parser.add_argument("--use_stable_loss", default=True, type=bool,
                       help="Use numerically stable loss function")
    parser.add_argument("--loss_gamma", default=0.5, type=float,
                       help="Loss gamma parameter")
    parser.add_argument("--loss_C4", default=10.0, type=float,
                       help="Loss C4 parameter")
    parser.add_argument("--loss_eps", default=0.0, type=float,
                       help="Loss epsilon for numerical stability")
    parser.add_argument("--clamp_predictions", default=False, type=bool,
                       help="Clamp predictions for stability")
    
    # W&B parameters
    parser.add_argument("--wandb_project", default="percepnet-enhanced", type=str,
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None, type=str,
                       help="W&B run name")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable W&B logging")
    
    # System parameters
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                       help="Rank for distributed training")

    args = parser.parse_args()

    # Device setup
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        logging.info("Using CPU for training")
    else:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        logging.info(f"Using GPU {args.rank} for training")

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Update config with command line arguments
    config.update(vars(args))
    config["use_wandb"] = WANDB_AVAILABLE and not args.disable_wandb
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.out_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
        
    # Log configuration
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # Initialize model
    model = PercepNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.0001))
    
    # Initialize enhanced loss function
    if args.use_stable_loss:
        criterion = CustomLossStable(
            gamma=args.loss_gamma,
            C4=args.loss_C4,
            eps=args.loss_eps,
            clamp_preds=args.clamp_predictions
        )
        logging.info("Using CustomLossStable with enhanced diagnostics")
    else:
        # Fallback to original loss (not recommended)
        from rnn_train import CustomLoss
        criterion = CustomLoss()
        logging.warning("Using original CustomLoss (may have NaN issues)")

    # Load datasets
    train_dataset = CppRawListDataset(
        args.train_filelist_path, train_length_size=args.train_length_size
    )
    dev_dataset = CppRawListDataset(
        args.dev_filelist_path, train_length_size=args.train_length_size
    )

    logging.info(f"Training files: {len(train_dataset)}")
    logging.info(f"Development files: {len(dev_dataset)}")

    # Create data loaders
    data_loader = {
        "train": DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4),
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        ),
        "dev": DataLoader(
            dev_dataset,
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4),
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
    }

    # Initialize enhanced trainer
    trainer = EnhancedTrainer(
        steps=0,
        epochs=0,
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        args=args,
        device=device,
    )

    # Load pretrained checkpoint if provided
    if args.pretrain:
        try:
            state_dict = torch.load(args.pretrain, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logging.info(f"Loaded pretrained model from {args.pretrain}")
        except Exception as e:
            logging.error(f"Failed to load pretrain checkpoint: {e}")
            raise

    # Start training
    try:
        trainer.run()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        # Save final checkpoint
        final_checkpoint = os.path.join(config["out_dir"], f"final-checkpoint-{trainer.steps}steps.pt")
        torch.save(model.state_dict(), final_checkpoint)
        logging.info(f"Saved final checkpoint: {final_checkpoint}")


if __name__ == '__main__':
    main()