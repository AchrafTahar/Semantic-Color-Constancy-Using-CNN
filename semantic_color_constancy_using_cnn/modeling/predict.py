from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from model import TrueColorNet
import typer
from loguru import logger
from tqdm import tqdm

from semantic_color_constancy_using_cnn.config import MODELS_DIR

app = typer.Typer()

# Initialize logger with file sink
logger.add("prediction.log", format="{time} {level} {message}", level="DEBUG")


class TrueColorNetPredictor:
    def __init__(self, model_path: Path):
        """Initialize the predictor with a trained model."""
        logger.info("Initializing the TrueColorNet predictor.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load model
            self.model = TrueColorNet().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.success(f"Model loaded successfully from {model_path}.")
        except Exception as e:
            logger.exception(f"Failed to load model from {model_path}: {e}")
            raise

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        logger.info("Transforms initialized.")

    def preprocess_image(self, image: Image.Image, mask: Image.Image) -> torch.Tensor:
        """Preprocess image and mask for the model."""
        logger.debug("Preprocessing image and mask.")
        img_tensor = self.transform(image)
        mask_tensor = self.transform(mask)

        # Combine image and mask
        x = torch.cat([img_tensor, mask_tensor], dim=0)
        x = x.unsqueeze(0)

        # Pixel-wise normalization
        mean = x[:, :3].mean(dim=(2, 3), keepdim=True)
        std = x[:, :3].std(dim=(2, 3), keepdim=True)
        x[:, :3] = (x[:, :3] - mean) / (std + 1e-7)

        logger.debug("Preprocessing completed.")
        return x

    def correct_white_balance(self, image_tensor: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply white balance correction using predicted parameters."""
        logger.debug("Applying white balance correction.")
        r, g, b, gamma = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]

        # Create correction matrix
        correction = torch.stack([1/r, 1/g, 1/b], dim=1).view(-1, 3, 1, 1)

        # Apply color correction and gamma correction
        corrected = image_tensor * correction
        corrected = torch.pow(corrected, 1 / gamma.view(-1, 1, 1, 1))

        logger.debug("White balance correction applied.")
        return corrected

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        logger.debug("Converting tensor to PIL Image.")
        tensor = torch.clamp(tensor, 0, 1)
        array = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)

    def predict(self, image_path: Path, mask_path: Path) -> (Image.Image, np.ndarray):
        """Predict and apply white balance correction."""
        logger.info(f"Starting prediction for image: {image_path} and mask: {mask_path}")
        try:
            # Load images
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # Preprocess
            x = self.preprocess_image(image, mask).to(self.device)
            original_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Predict parameters
            with torch.no_grad():
                params = self.model(x)

            # Apply correction
            corrected = self.correct_white_balance(original_tensor, params)
            corrected_image = self.tensor_to_image(corrected)

            logger.success("Prediction and correction completed successfully.")
            return corrected_image, params.squeeze().cpu().numpy()
        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            raise


@app.command()
def main(
    model_path: Path = typer.Argument(..., help="Path to the trained model checkpoint."),
    image_path: Path = typer.Argument(..., help="Path to the input image."),
    mask_path: Path = typer.Argument(..., help="Path to the semantic mask."),
    output_path: Path = typer.Argument(..., help="Path to save the corrected image."),
):
    """CLI tool for predicting and correcting images using TrueColorNet."""
    logger.info("CLI tool initialized.")
    try:
        predictor = TrueColorNetPredictor(model_path)

        corrected_image, params = predictor.predict(image_path, mask_path)

        logger.info("Saving corrected image.")
        corrected_image.save(output_path)
        logger.success(f"Corrected image saved to {output_path}.")

        logger.info(f"Correction Parameters:")
        logger.info(f"R multiplier: {1/params[0]:.3f}")
        logger.info(f"G multiplier: {1/params[1]:.3f}")
        logger.info(f"B multiplier: {1/params[2]:.3f}")
        logger.info(f"Gamma: {params[3]:.3f}")
    except Exception as e:
        logger.exception("An error occurred during processing.")


if __name__ == "__main__":
    app()