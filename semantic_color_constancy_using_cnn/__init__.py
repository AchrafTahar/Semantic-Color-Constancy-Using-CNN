from semantic_color_constancy_using_cnn import config  # noqa: F401
# Expose key functions/classes
from .config import MODELS_DIR, PROCESSED_DATA_DIR
from .modeling.model import TrueColorNet

# Optional metadata
__version__ = "1.0.0"

# Allow importing these directly from the package
__all__ = ["MODELS_DIR", "PROCESSED_DATA_DIR", "TrueColorNet"]
