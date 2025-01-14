from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
from . import RAW_DATA_DIR_IMG, RAW_DATA_DIR_MASK

app = typer.Typer()

# Configure logger to output to a file and console
logger.add("dataset_processing.log", format="{time} {level} {message}", level="DEBUG")


class ADE20KTrueColorNetDataset(Dataset):
    def __init__(self, root_dir_img: Path, root_dir_mask: Path, transform=None, train: bool = True):
        """
        Args:
            root_dir_img: Path to ADE20K dataset/images
            root_dir_mask: Path to ADE20K dataset/annotations
            transform: Optional transform to be applied
            train: If True, creates synthetic data for training
        """
        logger.info(f"Initializing ADE20K dataset from {root_dir_img} and {root_dir_mask}...")
        self.root_dir_img = root_dir_img
        self.root_dir_mask = root_dir_mask
        self.transform = transform
        self.train = train

        try:
            # Set paths for images and masks
            self.images_dir = root_dir_img / "training" if train else root_dir_img / "validation"
            self.masks_dir = root_dir_mask / "training" if train else root_dir_mask / "validation"

            # Get list of image files
            self.image_files = sorted(os.listdir(self.images_dir))

            if not self.image_files:
                raise ValueError("No image files found in the dataset directory.")

            # Set number of augmentations per image
            self.augmentations_per_image = 769 if train else 1
            logger.success(f"Dataset initialized successfully with {len(self.image_files)} images.")
        except Exception as e:
            logger.exception(f"Error initializing dataset: {e}")
            raise

    def __len__(self):
        return len(self.image_files) * self.augmentations_per_image

    def apply_wrong_white_balance(self, image: Image.Image):
        """Apply random white balance and gamma correction."""
        logger.debug("Applying random white balance and gamma correction.")
        img_array = np.array(image).astype(np.float32) / 255.0

        # Random RGB multipliers [0.7, 1.3]
        r, g, b = np.random.uniform(0.7, 1.3, 3)

        # Random gamma [0.85, 1.15]
        gamma = np.random.uniform(0.85, 1.15)

        # Apply color and gamma correction
        img_array[..., 0] *= r
        img_array[..., 1] *= g
        img_array[..., 2] *= b
        img_array = np.power(img_array, gamma)
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

        logger.debug("White balance correction applied.")
        return Image.fromarray(img_array), torch.tensor([r, g, b, gamma])

    def __getitem__(self, idx: int):
        """Retrieve a single dataset item."""
        img_idx = idx // self.augmentations_per_image
        aug_idx = idx % self.augmentations_per_image

        img_name = self.image_files[img_idx]
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / img_name.replace('.jpg', '.png')

        logger.debug(f"Loading image: {img_path}")
        logger.debug(f"Loading mask: {mask_path}")

        # Check if mask file exists
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply augmentations if training
        if self.train and aug_idx > 0:
            image, params = self.apply_wrong_white_balance(image)
        else:
            params = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Combine image and mask
        x = torch.cat([image, mask], dim=0)

        logger.debug(f"Returning augmented image and parameters for index {idx}.")
        return x, params


@app.command()
def main(
    root_dir_img: Path = typer.Argument(RAW_DATA_DIR_IMG / "training", help="Path to the ADE20K dataset root directory."),
    root_dir_mask: Path = typer.Argument(RAW_DATA_DIR_MASK / "training", help="Path to the ADE20K dataset root directory."),
    batch_size: int = typer.Option(32, help="Batch size for data loading."),
):
    """CLI tool to process the ADE20K dataset and generate features."""
    logger.info("Starting dataset processing...")

    try:
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Initialize dataset
        dataset = ADE20KTrueColorNetDataset(
            root_dir_img=root_dir_img,
            root_dir_mask=root_dir_mask,
            transform=transform,
            train=True
        )

        # Initialize dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Process dataset
        logger.info("Processing dataset...")
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Processing batches")):
            logger.debug(f"Batch {i + 1}: Inputs shape = {inputs.shape}, Targets shape = {targets.shape}")
            # Add your feature generation code here (if needed)

        logger.success("Dataset processing completed successfully.")
    except Exception as e:
        logger.exception("An error occurred during dataset processing.")
        raise


if __name__ == "__main__":
    app()