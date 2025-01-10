from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from model import TrueColorNet
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from semantic_color_constancy_using_cnn.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, RAW_DATA_DIR_IMG, RAW_DATA_DIR_MASK

app = typer.Typer()

# Initialize logger with a file sink
logger.add("training.log", format="{time} {level} {message}", level="DEBUG")

def initialize_weights(model):
    """Initialize model weights with pretrained AlexNet and Gaussian random for FC layers."""
    logger.info("Initializing weights using pretrained AlexNet.")
    alexnet = models.alexnet(pretrained=True)
    alexnet_conv1_weights = alexnet.features[0].weight.data
    
    with torch.no_grad():
        # First 3 dimensions (RGB) from AlexNet
        model.conv1.weight.data[:, :3, :, :] = alexnet_conv1_weights
        # Last dimension (mask) with average filter
        model.conv1.weight.data[:, 3, :, :] = 1.0 / 11.0

        # Initialize conv2-5 with pretrained weights
        model.conv2.weight.data = alexnet.features[3].weight.data
        model.conv3.weight.data = alexnet.features[6].weight.data
        model.conv4.weight.data = alexnet.features[8].weight.data
        model.conv5.weight.data = alexnet.features[10].weight.data

        # Initialize fc layers with random Gaussian weights
        for layer in [model.fc6, model.fc7, model.fc8, model.fc9]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias.data, 0.0)
    logger.success("Weights initialized successfully.")

def pixel_wise_normalization(images):
    """Apply pixel-wise normalization to RGB channels of the image tensor."""
    mean = images[:, :3].mean(dim=(2, 3), keepdim=True)
    std = images[:, :3].std(dim=(2, 3), keepdim=True)
    images[:, :3] = (images[:, :3] - mean) / (std + 1e-7)
    return images

def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Adjust learning rate based on the epoch."""
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.debug(f"Adjusted learning rate to {lr} at epoch {epoch}.")

def train(train_loader, val_loader, num_epochs=500):
    """Train the model with the given DataLoaders."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrueColorNet().to(device)
    
    initialize_weights(model)
    
    base_lr = 1e-5
    conv_params = list(model.conv1.parameters())
    new_params = (
        list(model.fc6.parameters()) +
        list(model.fc7.parameters()) +
        list(model.fc8.parameters()) +
        list(model.fc9.parameters())
    )
    optimizer = optim.SGD([
        {'params': conv_params, 'lr': base_lr},
        {'params': new_params, 'lr': base_lr * 50}
    ], momentum=0.95)

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch, base_lr)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = pixel_wise_normalization(data)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 99:
                logger.info(f"[Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = pixel_wise_normalization(data)
                output = model(data)
                val_loss += criterion(output, target).item()

        logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss / len(val_loader):.3f}")

        if (epoch + 1) % 50 == 0:
            checkpoint_path = MODELS_DIR / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """Main function to train the model."""
    logger.info("Starting model training...")
    try:
        # Load your data here (not implemented in this snippet)
        train_dataset = ADE20KTrueColorNetDataset(
        root_dir_img='RAW_DATA_DIR_IMG/training',
        root_dir_mask='RAW_DATA_DIR_MASK/training',
        transform=transform,
        train=True
        )
        val_dataset = ADE20KTrueColorNetDataset(
        root_dir_img='RAW_DATA_DIR_IMG/validation',
        root_dir_mask='RAW_DATA_DIR_MASK/training',
        transform=transform,
        train=False
        )
        # Create dataloaders
        train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        )
        val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
        )
        train(train_loader, val_loader)
        logger.success("Model training completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
