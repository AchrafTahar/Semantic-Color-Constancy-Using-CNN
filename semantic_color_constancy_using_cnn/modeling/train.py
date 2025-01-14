from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from semantic_color_constancy_using_cnn.config import MODELS_DIR, RAW_DATA_DIR_IMG, RAW_DATA_DIR_MASK
from semantic_color_constancy_using_cnn.dataset_processing import ADE20KTrueColorNetDataset
from semantic_color_constancy_using_cnn.modeling.model import TrueColorNet

app = typer.Typer()

# Initialize logger with a file sink
logger.add("training.log", format="{time} {level} {message}", level="DEBUG")

def initialize_weights(model):
    """Initialize model weights with pretrained AlexNet and Gaussian random for FC layers."""
    logger.info("Initializing weights using pretrained AlexNet.")
    alexnet = models.alexnet(pretrained=True)
    alexnet_conv1_weights = alexnet.features[0].weight.data
    
    with torch.no_grad():
        # First 64 channels from AlexNet
        model.conv1.weight.data[:64, :3, :, :] = alexnet_conv1_weights[:, :3, :, :]

        # Last channel (mask) initialized with average filter
        model.conv1.weight.data[:64, 3, :, :] = 1.0 / 11.0

        # Randomly initialize the remaining 32 channels
        if model.conv1.weight.data.size(0) > 64:
            nn.init.normal_(model.conv1.weight.data[64:], mean=0.0, std=0.01)

        # Initialize conv2 with pretrained weights from AlexNet (sliced to match 48 input channels)
        alexnet_conv2_weights = alexnet.features[3].weight.data
        alexnet_conv2_bias = alexnet.features[3].bias.data

        # Slice the weights to match 48 input channels
        model.conv2.weight.data[:192, :48, :, :] = alexnet_conv2_weights[:, :48, :, :]

        # Randomly initialize the remaining weights for conv2 (to match 256 output channels)
        if model.conv2.weight.data.size(0) > 192:
            nn.init.normal_(model.conv2.weight.data[192:], mean=0.0, std=0.01)

        # Initialize bias for conv2 (to match 256 output channels)
        model.conv2.bias.data[:192] = alexnet_conv2_bias
        if model.conv2.bias.data.size(0) > 192:
            nn.init.constant_(model.conv2.bias.data[192:], 0.0)

        # Initialize conv3 with random weights (to match 256 input channels)
        nn.init.normal_(model.conv3.weight.data, mean=0.0, std=0.01)
        nn.init.constant_(model.conv3.bias.data, 0.0)
        

        # Initialize conv4-5 with pretrained weights from AlexNet
        model.conv4.weight.data = alexnet.features[8].weight.data
        model.conv5.weight.data = alexnet.features[10].weight.data

        # Initialize FC layers (fully connected layers) with random weights
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
    # Return the trained model
    return model

@app.command()
def main(
    model_path: Path = MODELS_DIR,
    batch_size: int = 32,
):
    """Main function to train the model."""
    logger.info("Starting model training...")
    try:
        # Load your data here (not implemented in this snippet)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = ADE20KTrueColorNetDataset(
            root_dir_img=RAW_DATA_DIR_IMG,
            root_dir_mask=RAW_DATA_DIR_MASK,
            transform=transform,
            train=True
        )
        val_dataset = ADE20KTrueColorNetDataset(
            root_dir_img=RAW_DATA_DIR_IMG,
            root_dir_mask=RAW_DATA_DIR_MASK,
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
        model = train(train_loader, val_loader)

        # Save the model after training
        torch.save(model.state_dict(), model_path)
        logger.success("Model training completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    app()