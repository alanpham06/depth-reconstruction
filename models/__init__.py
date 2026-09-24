"""The network, and nothing else.

Kept clear of Lightning, the loss and the dataset, so that export/export.py can
import it into the executorch conda env, which has neither Lightning nor
matplotlib.
"""

from models.unet import UNet

__all__ = ["UNet"]
