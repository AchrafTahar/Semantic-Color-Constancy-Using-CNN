from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from semantic_color_constancy_using_cnn.config import PROCESSED_DATA_DIR

app = typer.Typer()
def correct_white_balance(image, params):
    """
    Apply white balance correction using predicted parameters
    Args:
        image: Input image tensor (B, 3, H, W)
        params: Predicted parameters tensor (B, 4) [r, g, b, gamma]
    """
    r, g, b, gamma = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
    
    # Create correction matrix
    correction = torch.stack([1/r, 1/g, 1/b], dim=1)
    correction = correction.view(-1, 3, 1, 1)
    
    # Apply color correction
    corrected = image * correction
    
    # Apply gamma correction
    corrected = torch.pow(corrected, 1/gamma.view(-1, 1, 1, 1))
    
    return corrected

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
