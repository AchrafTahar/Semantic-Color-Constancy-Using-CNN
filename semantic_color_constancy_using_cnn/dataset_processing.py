from pathlib import Path
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
from semantic_color_constancy_using_cnn.config import RAW_DATA_DIR_IMG, OUT_DATA_DIR_IMG

app = typer.Typer()

# Configure logger to output to a file and console
logger.add("dataset_processing.log", format="{time} {level} {message}", level="DEBUG", rotation="10 MB", retention="10 days", compression="zip")

class ADE20KTrueColorNetDataset(Dataset):
    def __init__(self, root_dir_img: Path, output_dir_img: Path, transform=None, split_type: str = "training"):
        """
        Args:
            root_dir_img: Path to ADE20K dataset/images
            output_dir_img: Path to output directory for processed images
            transform: Optional transform to be applied
            split_type: One of "training", "validation", or "test"
        """
        logger.info(f"Initializing ADE20K dataset from {root_dir_img} for {split_type} split...")
        self.root_dir_img = root_dir_img
        self.output_dir_img = output_dir_img
        self.transform = transform
        self.split_type = split_type

        try:
            # For training set, use the original training directory
            if split_type == "training":
                self.images_root_dir = root_dir_img / "training"
                self.images_out_dir = output_dir_img / "training"
            # For validation and test, both will use the original validation directory
            else:
                self.images_root_dir = root_dir_img / "validation"
                self.images_out_dir = output_dir_img / split_type
            
            # Create output directory if it doesn't exist
            self.images_out_dir.mkdir(parents=True, exist_ok=True)

            # Get all image files
            all_image_files = sorted(os.listdir(self.images_root_dir))
            if not all_image_files:
                raise ValueError("No image files found in the dataset directory.")
                
            # If we're using validation or test, we need to determine which files to use
            if split_type != "training":
                # Get the full list of validation files
                val_files = all_image_files
                
                # Set a random seed for reproducibility
                np.random.seed(42)
                
                # Shuffle the files
                np.random.shuffle(val_files)
                
                # Split the files: 50% for validation, 50% for test
                split_idx = len(val_files) // 2
                
                if split_type == "validation":
                    self.image_files = val_files[:split_idx]
                else:  # test set
                    self.image_files = val_files[split_idx:]
            else:
                # Use all files for training
                self.image_files = all_image_files

            logger.success(f"{split_type.capitalize()} dataset initialized successfully with {len(self.image_files)} images.")
        except Exception as e:
            logger.exception(f"Error initializing dataset: {e}")
            raise

    def __len__(self):
        return len(self.image_files)

    def apply_wrong_white_balance(self, image: Image.Image):
        """Apply random white balance and gamma correction."""
        logger.debug("Applying random white balance and gamma correction.")
        img_array = np.array(image).astype(np.float32) / 255.0

        r, g, b, gamma = np.random.uniform([0.7, 0.7, 0.7, 0.85], [1.3, 1.3, 1.3, 1.15])

        img_array[..., 0] *= r
        img_array[..., 1] *= g
        img_array[..., 2] *= b
        img_array = np.power(img_array, gamma)
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

        logger.debug("White balance and gamma correction applied.")
        return Image.fromarray(img_array), torch.tensor([r, g, b, gamma])

    def normalize_image(self, image: torch.Tensor):
        """Apply pixel-wise normalization."""
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True) + 1e-6
        return (image - mean) / std

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = self.images_root_dir / img_name

        image = Image.open(img_path).convert('RGB')

        image, params = self.apply_wrong_white_balance(image)

        image_save_path = self.images_out_dir / f"{img_name.replace('.jpg', f'.png')}"
        image.save(image_save_path)
        logger.info(f"Saved processed image to {image_save_path}")

        if self.transform:
            image = self.transform(image)

        image = self.normalize_image(image)

        # Return the image and the white balance parameters
        return image, params
    
def compute_inverse_params(params):
    """Compute the inverse of the white balance and gamma correction parameters."""
    r, g, b, gamma = params
    inv_r = 1.0 / r
    inv_g = 1.0 / g
    inv_b = 1.0 / b
    inv_gamma = 1.0 / gamma
    return inv_r, inv_g, inv_b, inv_gamma

def create_output_directories(output_dir_img: Path):
    """Create output directories if they don't exist."""
    (output_dir_img / "training").mkdir(parents=True, exist_ok=True)
    (output_dir_img / "validation").mkdir(parents=True, exist_ok=True)
    (output_dir_img / "test").mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directories at {output_dir_img}")

@app.command()
def main(
    root_dir_img: Path = typer.Argument(RAW_DATA_DIR_IMG, help="Path to the ADE20K dataset root directory for images."),
    output_dir_img: Path = typer.Argument(OUT_DATA_DIR_IMG, help="Path to the ADE20K dataset output directory for images."),
    batch_size: int = typer.Option(32, help="Batch size for data loading."),
):
    """CLI tool to process the ADE20K dataset and generate features."""
    logger.info("Starting dataset processing...")

    try:
        # Create output directories
        create_output_directories(output_dir_img)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Process training set
        train_dataset = ADE20KTrueColorNetDataset(
            root_dir_img=root_dir_img,
            output_dir_img=output_dir_img,
            transform=transform,
            split_type="training"
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        logger.info("Processing training dataset...")
        for i, (inputs, targets) in enumerate(tqdm(train_dataloader, desc="Processing training batches")):
            logger.debug(f"Batch {i + 1}: Inputs shape = {inputs.shape}, Targets shape = {targets.shape}")

        logger.success("Training dataset processing completed successfully.")

        # Process validation set
        val_dataset = ADE20KTrueColorNetDataset(
            root_dir_img=root_dir_img,
            output_dir_img=output_dir_img,
            transform=transform,
            split_type="validation"
        )

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info("Processing validation dataset...")
        # Open CSV file to save validation inverse parameters
        with open(output_dir_img / 'inverse_params_val.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'inv_r', 'inv_g', 'inv_b', 'inv_gamma'])

            for i, (inputs, targets) in enumerate(tqdm(val_dataloader, desc="Processing validation batches")):
                logger.debug(f"Batch {i + 1}: Inputs shape = {inputs.shape}, Targets shape = {targets.shape}")
                for j, params in enumerate(targets):
                    if i * batch_size + j < len(val_dataset):
                        img_name = val_dataset.image_files[i * batch_size + j]
                        inv_params = compute_inverse_params(params.numpy())
                        writer.writerow([img_name] + list(inv_params))

        logger.success("Validation dataset processing completed successfully.")
        
        # Process test set
        test_dataset = ADE20KTrueColorNetDataset(
            root_dir_img=root_dir_img,
            output_dir_img=output_dir_img,
            transform=transform,
            split_type="test"
        )

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info("Processing test dataset...")
        # Open CSV file to save test inverse parameters
        with open(output_dir_img / 'inverse_params_test.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'inv_r', 'inv_g', 'inv_b', 'inv_gamma'])

            for i, (inputs, targets) in enumerate(tqdm(test_dataloader, desc="Processing test batches")):
                logger.debug(f"Batch {i + 1}: Inputs shape = {inputs.shape}, Targets shape = {targets.shape}")
                for j, params in enumerate(targets):
                    if i * batch_size + j < len(test_dataset):
                        img_name = test_dataset.image_files[i * batch_size + j]
                        inv_params = compute_inverse_params(params.numpy())
                        writer.writerow([img_name] + list(inv_params))

        logger.success("Test dataset processing completed successfully.")
        
    except Exception as e:
        logger.exception("An error occurred during dataset processing.")
        raise

if __name__ == "__main__":
    app()