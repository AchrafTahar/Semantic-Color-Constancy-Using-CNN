from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
from semantic_color_constancy_using_cnn.config import RAW_DATA_DIR_IMG, RAW_DATA_DIR_MASK, OUT_DATA_DIR_IMG, OUT_DATA_DIR_MASK

app = typer.Typer()

# Configure logger to output to a file and console
logger.add("dataset_processing.log", format="{time} {level} {message}", level="DEBUG")

class ADE20KTrueColorNetDataset(Dataset):
    def __init__(self, root_dir_img: Path, root_dir_mask: Path, output_dir_img: Path, output_dir_mask: Path, transform=None, train: bool = True):
        """
        Args:
            root_dir_img: Path to ADE20K dataset/images
            root_dir_mask: Path to ADE20K dataset/annotations
            output_dir: Path to save processed images and masks
            transform: Optional transform to be applied
            train: If True, creates synthetic data for training
        """
        logger.info(f"Initializing ADE20K dataset from {root_dir_img} and {root_dir_mask}...")
        self.root_dir_img = root_dir_img
        self.root_dir_mask = root_dir_mask
        self.output_dir_img = output_dir_img
        self.output_dir_mask = output_dir_mask
        self.transform = transform
        self.train = train

        try:
            self.images_root_dir = root_dir_img / "training" if train else root_dir_img / "validation"
            self.masks_root_dir = root_dir_mask / "training" if train else root_dir_mask / "validation"
            self.images_out_dir = output_dir_img / "training" if train else output_dir_img / "validation"
            self.masks_out_dir = output_dir_mask / "training" if train else output_dir_mask / "validation"

            self.image_files = sorted(os.listdir(self.images_root_dir))
            if not self.image_files:
                raise ValueError("No image files found in the dataset directory.")

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

        r, g, b = np.random.uniform(0.7, 1.3, 3)
        gamma = np.random.uniform(0.85, 1.15)

        img_array[..., 0] *= r
        img_array[..., 1] *= g
        img_array[..., 2] *= b
        img_array = np.power(img_array, gamma)
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

        logger.debug("White balance and gamma correction applied.")
        return Image.fromarray(img_array), torch.tensor([r, g, b, gamma])

    def apply_spatial_augmentation(self, image: Image.Image, mask: Image.Image):
        """Apply the same spatial augmentation to both the image and mask."""
        if np.random.rand() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)
        if np.random.rand() > 0.5:
            translate = (np.random.uniform(-0.1, 0.1) * image.size[0], np.random.uniform(-0.1, 0.1) * image.size[1])
            scale = np.random.uniform(0.9, 1.1)
            image = F.affine(image, angle=0, translate=translate, scale=scale, shear=0)
            mask = F.affine(mask, angle=0, translate=translate, scale=scale, shear=0)
        return image, mask

    def normalize_image(self, image: torch.Tensor):
        """Apply pixel-wise normalization."""
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True) + 1e-6  # Avoid division by zero
        return (image - mean) / std

    def __getitem__(self, idx: int):
        """Retrieve a single dataset item."""
        img_idx = idx // self.augmentations_per_image
        aug_idx = idx % self.augmentations_per_image

        img_name = self.image_files[img_idx]
        img_path = self.images_root_dir / img_name
        mask_path = self.masks_root_dir / img_name.replace('.jpg', '.png')

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.train and aug_idx > 0:
            image, params = self.apply_wrong_white_balance(image)
            image, mask = self.apply_spatial_augmentation(image, mask)
        else:
            params = torch.tensor([1.0, 1.0, 1.0, 1.0])

        image_save_path = self.images_out_dir / f"{img_name.replace('.jpg', f'_{aug_idx}.png')}"
        mask_save_path = self.masks_out_dir / f"{img_name.replace('.jpg', f'_mask_{aug_idx}.png')}"
        image.save(image_save_path)
        mask.save(mask_save_path)
        logger.info(f"Saved processed image to {image_save_path}")
        logger.info(f"Saved processed mask to {mask_save_path}")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = self.normalize_image(image)

        x = torch.cat([image, mask], dim=0)
        return x, params

@app.command()
def main(
    root_dir_img: Path = typer.Argument(RAW_DATA_DIR_IMG, help="Path to the ADE20K dataset root directory for images."),
    root_dir_mask: Path = typer.Argument(RAW_DATA_DIR_MASK, help="Path to the ADE20K dataset root directory for masks."),
    output_dir_img: Path = typer.Argument(OUT_DATA_DIR_IMG, help="Path to the ADE20K dataset output directory for images."),
    output_dir_mask: Path = typer.Argument(OUT_DATA_DIR_MASK, help="Path to the ADE20K dataset output directory for masks."),
    batch_size: int = typer.Option(32, help="Batch size for data loading."),
):
    """CLI tool to process the ADE20K dataset and generate features."""
    logger.info("Starting dataset processing...")

    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = ADE20KTrueColorNetDataset(
            root_dir_img=root_dir_img,
            root_dir_mask=root_dir_mask,
            output_dir_img=output_dir_img,
            output_dir_mask=output_dir_mask,
            transform=transform,
            train=True
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        logger.info("Processing dataset...")
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Processing batches")):
            logger.debug(f"Batch {i + 1}: Inputs shape = {inputs.shape}, Targets shape = {targets.shape}")

        logger.success("Dataset processing completed successfully.")
    except Exception as e:
        logger.exception("An error occurred during dataset processing.")
        raise

if __name__ == "__main__":
    app()
