"""POST3R: Persistent Object Slots for Temporal 3D Representation from Video"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models import (
    TTT3RBackbone,
    RecurrentSlotAttention,
    Decoder3D,
    POST3R,
)

from .losses import (
    PointmapReconstructionLoss,
    POST3RLoss,
)

__all__ = [
    # Models
    'TTT3RBackbone',
    'RecurrentSlotAttention',
    'Decoder3D',
    'POST3R',
    # Losses
    'PointmapReconstructionLoss',
    'TemporalConsistencyLoss',
    'SlotRegularizationLoss',
    'POST3RLoss',
]
