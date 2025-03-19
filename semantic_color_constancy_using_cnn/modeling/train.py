from pathlib import Path
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm
from PIL import Image  # Add this import
from semantic_color_constancy_using_cnn.config import MODELS_DIR, RAW_DATA_DIR_IMG, RAW_DATA_DIR_MASK, OUT_DATA_DIR_IMG
from semantic_color_constancy_using_cnn.modeling.model import TrueColorNet


app = typer.Typer()

# Initialize logger with a file sink
logger.add("training.log", format="{time} {level} {message}", level="DEBUG", rotation="5 MB", retention="5 days", compression="zip") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColorConstancyDataset(Dataset):
    def __init__(self, root_dir_img, root_dir_mask, ground_truth_dir, transform=None):
        """
        Args:
            root_dir_img (str): Directory with input images
            root_dir_mask (str): Directory with semantic masks
            ground_truth_dir (str): Directory with illumination ground truth
            transform (callable): Transform to be applied on images
            train (bool): If True, creates dataset from training set
        """
        self.root_dir_img = root_dir_img
        self.root_dir_mask = root_dir_mask
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        
        # Get list of all images
        self.images = [f for f in os.listdir(root_dir_img) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir_img, img_name)
        mask_path = os.path.join(self.root_dir_mask, img_name.replace('.jpg', '_mask.png'))
        img_name = img_name.replace('png', 'jpg')
        target_path = os.path.join(self.ground_truth_dir, img_name)
        
        
        # Load image, mask and target
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  
        target = Image.open(target_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            target = self.transform(target)
        
        
        # Combine image and mask
        combined_input = torch.cat([image, mask], dim=0)
        
        return combined_input, target
    
def initialize_weights(model):
    """Initialize model weights with pretrained AlexNet and Gaussian random for FC layers."""
    logger.info("Initializing weights using pretrained AlexNet.")
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    alexnet_conv1_weights = alexnet.features[0].weight.data
    
    with torch.no_grad():
        # Initialize conv1 weights
        # Can't directly resize tensors with different shapes
        # Initialize with random values first
        nn.init.normal_(model.conv1.weight.data, mean=0.0, std=0.001)
        
        # For the first 3 channels (RGB), copy weights from AlexNet for the first 64 filters
        # and initialize the rest randomly
        model.conv1.weight.data[:64, :3, :, :] = alexnet_conv1_weights
        
        # For the 4th channel (mask), use average of RGB filters for first 64 filters
        mask_weights = torch.mean(alexnet_conv1_weights, dim=1, keepdim=True)
        model.conv1.weight.data[:64, 3:4, :, :] = mask_weights
        
        # Initialize conv2-conv5 with random weights
        # We can't use pretrained weights directly because the dimensions don't match
        for layer in [model.conv2, model.conv3, model.conv4, model.conv5]:
            nn.init.kaiming_normal_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0.0)

        # Initialize FC layers with random Gaussian weights
        for layer in [model.fc6, model.fc7, model.fc8, model.fc9]:
            nn.init.normal_(layer.weight.data, mean=0.0, std=0.001)
            nn.init.constant_(layer.bias.data, 0.0)
            
    logger.success("Weights initialized successfully.")

def pixel_wise_normalization(images):
    """Apply pixel-wise normalization to RGB channels of the image tensor."""
    # First check for NaNs in input
    if torch.isnan(images).any():
        logger.error("NaN detected in images before normalization!")
        
    # Only normalize RGB channels, leave mask channel unchanged
    rgb = images[:, :3]
    mask = images[:, 3:4]
    
    # Compute stats per image, add small epsilon to avoid division by zero
    mean = rgb.mean(dim=(2, 3), keepdim=True)
    std = rgb.std(dim=(2, 3), keepdim=True) + 1e-5
    
    # Normalize RGB channels
    rgb_normalized = (rgb - mean) / std
    
    # Replace any NaN or Inf values with zeros
    rgb_normalized = torch.where(torch.isfinite(rgb_normalized), rgb_normalized, torch.zeros_like(rgb_normalized))
    
    # Log stats after normalization
    if torch.isnan(rgb_normalized).any():
        logger.error("NaN detected after normalization!")
    
    # Combine normalized RGB with unchanged mask
    result = torch.cat([rgb_normalized, mask], dim=1)
    
    return result

def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Adjust learning rate based on the epoch."""
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.debug(f"Adjusted learning rate to {lr} at epoch {epoch}.")

def train(train_loader, val_loader, num_epochs=500):
    model = TrueColorNet().to(device)
    initial_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data, target = data.to(device), target.to(device)
            data = pixel_wise_normalization(data)
            optimizer.zero_grad()

            output = model(data)
            
            if torch.isnan(output).any():
                logger.warning(f"NaN detected in model output at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()
            valid_batches += 1

            if batch_idx % 100 == 99:
                avg_loss = running_loss / valid_batches if valid_batches > 0 else float('nan')
                logger.info(f"[Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {avg_loss:.6f}")  
                running_loss = 0.0
                valid_batches = 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = pixel_wise_normalization(data)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_train_loss = running_loss / valid_batches if valid_batches > 0 else float('inf')
        
        # Save losses for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {epoch_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_path = MODELS_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
            
        # Early stopping check
        if patience_counter >= max_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Regular checkpoint saving
        if (epoch + 1) % 10 == 0:
            checkpoint_path = MODELS_DIR / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'current_loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(MODELS_DIR / 'training_history.png')
        logger.info(f"Training history plot saved to {MODELS_DIR / 'training_history.png'}")
    except ImportError:
        logger.warning("Matplotlib not available. Skipping loss plot generation.")
        
    # Load and return the best model
    checkpoint = torch.load(MODELS_DIR / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_losses, val_losses

@app.command()
def main(
    model_path: Path = MODELS_DIR,
    batch_size: int = 32,
):
    """Main function to train the model."""
    logger.info("Starting model training...")
    try:
        # Data transformations
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
        ])

        # Load training data
        train_dataset = ColorConstancyDataset(
            root_dir_img=OUT_DATA_DIR_IMG/"training",
            root_dir_mask=RAW_DATA_DIR_MASK/"training",
            ground_truth_dir=RAW_DATA_DIR_IMG/"training",  
            transform=transform,
        )

        # Load validation data
        val_dataset = ColorConstancyDataset(
            root_dir_img=OUT_DATA_DIR_IMG/"validation",
            root_dir_mask=RAW_DATA_DIR_MASK/"validation",
            ground_truth_dir=RAW_DATA_DIR_IMG/"validation",  
            transform=transform,
        )

        logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

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

        # Train the model
        model = train(train_loader, val_loader, 50)

        # Save the final model
        model_save_path = model_path / "final_model.pth"
        torch.save(model.state_dict(), model_save_path)
        logger.success(f"Model training completed successfully. Model saved to {model_save_path}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    app()