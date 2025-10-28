#!/usr/bin/env python3
"""
Training script for POST3R

Usage:
    python scripts/train.py configs/train/ytvis2021.yaml
    python scripts/train.py configs/train/ytvis2021.yaml --data-dir /path/to/data
    python scripts/train.py configs/train/ytvis2021.yaml --log-dir outputs learning_rate=1e-3

Output directory structure:
    outputs/
        <experiment_name>/
            <timestamp>/
                config.yaml          # Saved configuration
                logs/                # Training logs
                    tensorboard/     # TensorBoard logs
                    metrics/         # CSV metrics
                results/             # Training results
                    checkpoints/     # Model checkpoints
                visualizations/      # Visualization images/videos
"""

import argparse
import logging
import os
import pathlib
import warnings
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_info as log_info

from post3r.training import POST3RLightningModule

# Output directory structure
LOGS_SUBDIR = "logs"  # For training logs (tensorboard, metrics)
RESULTS_SUBDIR = "results"  # For checkpoints and final outputs
VISUALIZATIONS_SUBDIR = "visualizations"  # For visualization images/videos

# Subdirectories under logs/
TENSORBOARD_SUBDIR = "tensorboard"
METRICS_SUBDIR = "metrics"

# Subdirectories under results/
CHECKPOINT_SUBDIR = "checkpoints"

# Argument parser
parser = argparse.ArgumentParser(description="Train POST3R model")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
group.add_argument("-q", "--quiet", action="store_true", help="Suppress outputs")
parser.add_argument("-n", "--dry", action="store_true", help="Dry run (no logfiles)")
parser.add_argument("--no-tensorboard", action="store_true", help="Do not write tensorboard logs")
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--wandb-project", default="post3r", help="W&B project name")
parser.add_argument("--wandb-entity", default="zhiyuanli", help="W&B entity/team name")
parser.add_argument("--timeout", help="Stop training after this time (format: DD:HH:MM:SS)")
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--log-dir", default="outputs", help="Base directory for experiment outputs")
parser.add_argument(
    "--continue",
    dest="continue_from",
    type=pathlib.Path,
    help="Continue training from this log folder or checkpoint path",
)
parser.add_argument("config", help="Path to configuration YAML file")
parser.add_argument("config_overrides", nargs="*", help="Config overrides (key=value)")


def load_config(config_path: str, config_overrides: list = None):
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    
    # Apply command-line overrides
    if config_overrides:
        override_dict = {}
        for override in config_overrides:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to convert to appropriate type
                try:
                    value = eval(value)
                except:
                    pass  # Keep as string
                override_dict[key] = value
        
        config = OmegaConf.merge(config, OmegaConf.create(override_dict))
    
    return config


def setup_callbacks(args, config, output_path: pathlib.Path):
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    if not args.dry:
        # Model checkpoint - save periodically without monitoring a metric
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=output_path / RESULTS_SUBDIR / CHECKPOINT_SUBDIR,
            filename="{epoch:02d}-{step:06d}",
            save_last=True,  # Always save the last checkpoint
            every_n_train_steps=config.get('checkpoint_every_n_steps', 5000),
            verbose=args.verbose,
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # Progress bar with loss metrics
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    callbacks.append(progress_bar)
    
    # Timeout callback
    if args.timeout:
        timer = pl.callbacks.Timer(
            duration=args.timeout,
            interval="step",
            verbose=args.verbose
        )
        callbacks.append(timer)
    
    return callbacks


def setup_loggers(args, config, log_path: pathlib.Path):
    """Setup PyTorch Lightning loggers."""
    if args.dry:
        return []
    
    loggers = []
    
    # TensorBoard logger
    if not args.no_tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=log_path / LOGS_SUBDIR,
            name=TENSORBOARD_SUBDIR,
            version=""
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if args.wandb:
        try:
            import wandb
            
            # Get experiment name and timestamp from path
            experiment_name = config.get('experiment_name', 'post3r')
            timestamp = log_path.name
            run_name = f"{experiment_name}_{timestamp}"
            
            wandb_logger = pl.loggers.WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                save_dir=log_path / LOGS_SUBDIR,
                config=OmegaConf.to_container(config, resolve=True),
                log_model=False,  # Don't upload checkpoints automatically
            )
            loggers.append(wandb_logger)
            log_info(f"Weights & Biases logging enabled: {args.wandb_project}/{run_name}")
        except ImportError:
            log_info("Warning: wandb not installed. Skipping W&B logging.")
            log_info("Install with: pip install wandb")
    
    # CSV logger for metrics
    csv_logger = pl.loggers.CSVLogger(
        save_dir=log_path / LOGS_SUBDIR,
        name=METRICS_SUBDIR
    )
    loggers.append(csv_logger)
    
    return loggers


def setup_trainer_config(trainer_config: dict) -> dict:
    """Setup trainer configuration with sensible defaults."""
    # Training steps
    if "max_steps" not in trainer_config:
        trainer_config["max_steps"] = 100000
        log_info(f"Setting max_steps to {trainer_config['max_steps']}")
    
    # Remove max_epochs if present (we use max_steps)
    if "max_epochs" in trainer_config:
        del trainer_config["max_epochs"]
        log_info("Removing max_epochs (using max_steps instead)")
    
    # Validation frequency
    if "val_check_interval" not in trainer_config:
        trainer_config["val_check_interval"] = 5000
        log_info(f"Setting val_check_interval to {trainer_config['val_check_interval']}")
    
    # Logging frequency
    if "log_every_n_steps" not in trainer_config:
        trainer_config["log_every_n_steps"] = 100
        log_info(f"Setting log_every_n_steps to {trainer_config['log_every_n_steps']}")
    
    # Device selection
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "auto"
    
    # Distributed strategy
    if (
        "strategy" not in trainer_config
        and trainer_config.get("accelerator") != "cpu"
        and trainer_config.get("devices") != 1
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        trainer_config["strategy"] = "ddp"
        log_info("Setting strategy to ddp for multi-GPU training")
    
    # Gradient clipping
    if "gradient_clip_val" not in trainer_config:
        trainer_config["gradient_clip_val"] = 1.0
        log_info("Setting gradient_clip_val to 1.0")
    
    return trainer_config


def make_log_dir(base_dir: str, experiment_name: str) -> pathlib.Path:
    """Create log directory with timestamp."""
    from datetime import datetime
    
    base_path = pathlib.Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base_path / experiment_name / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    
    return log_path


def find_last_checkpoint(log_path: pathlib.Path) -> Optional[str]:
    """Find the last checkpoint in log directory."""
    checkpoint_dir = log_path / RESULTS_SUBDIR / CHECKPOINT_SUBDIR
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # Find last.ckpt or the most recent checkpoint
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
    
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    return str(checkpoints[-1])


def main():
    """Main training function."""
    args = parser.parse_args()
    
    # Optimize for AMD MI250X tensor cores (also works for NVIDIA)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # or 'medium' for more precision
    
    # Load configuration
    config = load_config(args.config, args.config_overrides)
    log_info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Setup logging
    if args.quiet:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning)
    elif not args.verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    
    # Setup log directory
    log_path: Optional[pathlib.Path] = None
    if not args.dry:
        if args.continue_from and args.continue_from.is_dir():
            log_path = args.continue_from
        else:
            experiment_name = config.get('experiment_name', 'post3r')
            log_path = make_log_dir(args.log_dir, experiment_name)
        
        log_info(f"Experiment directory: {log_path}")
        
        # Create subdirectories
        (log_path / LOGS_SUBDIR).mkdir(parents=True, exist_ok=True)
        (log_path / RESULTS_SUBDIR).mkdir(parents=True, exist_ok=True)
        (log_path / VISUALIZATIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_save_path = log_path / "config.yaml"
        OmegaConf.save(config, config_save_path)
        log_info(f"Saved config to {config_save_path}")
        log_info(f"  - Logs: {log_path / LOGS_SUBDIR}")
        log_info(f"  - Results: {log_path / RESULTS_SUBDIR}")
        log_info(f"  - Visualizations: {log_path / VISUALIZATIONS_SUBDIR}")
    
    # Find checkpoint for resuming
    ckpt_path: Optional[str] = None
    if args.continue_from:
        if args.continue_from.is_file() and args.continue_from.suffix == ".ckpt":
            ckpt_path = str(args.continue_from)
        elif args.continue_from.is_dir():
            ckpt_path = find_last_checkpoint(args.continue_from)
        
        if ckpt_path:
            log_info(f"Resuming from checkpoint: {ckpt_path}")
        else:
            log_info("No checkpoint found, starting from scratch")
    
    # Override data directory if provided
    data_dir_override = args.data_dir if args.data_dir else None
    
    # Create data module using SlotContrast-style build function
    log_info("Creating data module...")
    from post3r.training import data_module as dm
    data_module = dm.build(config.dataset, data_dir=data_dir_override)
    if args.verbose:
        log_info(str(data_module))
    
    # Build metrics (SlotContrast-style)
    log_info("Building metrics...")
    if config.get('train_metrics') is not None:
        from post3r.training import metrics as metrics_module
        train_metrics = {
            name: metrics_module.build(metric_config)
            for name, metric_config in config.train_metrics.items()
        }
        log_info(f"Train metrics: {list(train_metrics.keys())}")
    else:
        train_metrics = None
    
    if config.get('val_metrics') is not None:
        from post3r.training import metrics as metrics_module
        val_metrics = {
            name: metrics_module.build(metric_config)
            for name, metric_config in config.val_metrics.items()
        }
        log_info(f"Val metrics: {list(val_metrics.keys())}")
    else:
        val_metrics = None
    
    # Create model
    log_info("Creating model...")
    if ckpt_path:
        # Load from checkpoint
        model = POST3RLightningModule.load_from_checkpoint(ckpt_path)
    else:
        # Create new model with metrics
        model_config = dict(config.model)
        model_config['train_metrics'] = train_metrics
        model_config['val_metrics'] = val_metrics
        
        model = POST3RLightningModule(**model_config)
    
    log_info(f"\n{model}")
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(args, config, log_path) if log_path else []
    loggers = setup_loggers(args, config, log_path) if log_path else []
    
    # Setup trainer configuration
    trainer_config = setup_trainer_config(dict(config.get('trainer', {})))
    
    # Create trainer
    log_info("Creating trainer...")
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=not args.quiet,
    )
    
    # Train
    log_info("Starting training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    log_info("Training complete!")


if __name__ == "__main__":
    main()

