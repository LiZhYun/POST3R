"""
POST3R Data Module for PyTorch Lightning.

This module follows SlotContrast's approach using webdataset for efficient data loading.
"""
from typing import Optional

from post3r.data.datamodules import build as build_datamodule


def build(config, data_dir: Optional[str] = None):
    """Build data module from configuration.
    
    This function matches SlotContrast's data.build() interface.
    """
    return build_datamodule(config, data_dir=data_dir)
