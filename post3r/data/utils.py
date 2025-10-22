"""
Utility functions for data loading.
"""
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "./post3r/data"


def get_data_root_dir(error_on_missing_path: bool = False) -> Optional[str]:
    """Get the root directory for data.
    
    Checks POST3R_DATA_DIR environment variable first, then falls back to default.
    """
    data_dir = os.environ.get("POST3R_DATA_DIR")
    if data_dir and not os.path.isdir(data_dir) and error_on_missing_path:
        raise ValueError(f"Path {data_dir} specified by POST3R_DATA_DIR does not exist")

    if data_dir is None and os.path.isdir(DEFAULT_DATA_DIR):
        data_dir = DEFAULT_DATA_DIR

    if data_dir is None and error_on_missing_path:
        raise ValueError(
            f"No data root dir found. Check POST3R_DATA_DIR environment variable "
            f"or create {DEFAULT_DATA_DIR}"
        )

    return data_dir


def worker_init_function(worker_id: int, rank: Optional[int] = None) -> None:
    """Dataloader worker init function setting unique random seeds per worker.
    
    Based on PyTorch Lightning's implementation.
    """
    from pytorch_lightning.utilities import rank_zero_only

    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    # Back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    log.debug(
        f"Initializing random number generators of process {global_rank} worker {worker_id} "
        f"with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # Use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # Use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)


def remap_dict(sample: Dict[str, Any], rename_dict: Dict[str, str]) -> Dict[str, Any]:
    """Rename the keys of the dict."""
    for k, v in rename_dict.items():
        sample[v] = sample[k]
        del sample[k]
    return sample
