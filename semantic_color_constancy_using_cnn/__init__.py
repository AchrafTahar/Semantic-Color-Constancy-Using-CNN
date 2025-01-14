from semantic_color_constancy_using_cnn import config  # noqa: F401
# Expose key functions/classes
from .config import MODELS_DIR, RAW_DATA_DIR_IMG, RAW_DATA_DIR_MASK
from .modeling.model import TrueColorNet
from .dataset_processing import ADE20KTrueColorNetDataset

# Optional metadata
__version__ = "1.0.0"

# Allow importing these directly from the package
__all__ = ["MODELS_DIR", "RAW_DATA_DIR_IMG", "RAW_DATA_DIR_MASK", "TrueColorNet", "ADE20KTrueColorNetDataset"]
