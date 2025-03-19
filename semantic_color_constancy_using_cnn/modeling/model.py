# All the needed importations.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F


# Here we define The TrueColorNet neural network based on the AlexNet architecture and adapt it to recieve 4D volume as input. 
class TrueColorNet(nn.Module):
    def __init__(self):
        super(TrueColorNet, self).__init__()

        # Conv1: 96 11x11x4 convolutions with stride [4 4] and padding [0 0].
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv2: 256 5x5x48 convolutions with stride [1 1] and padding [2 2].
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv3: 384 3x3x256 convolutions with stride [1 1] and padding [1 1].
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)

        # Conv4: 384 3x3x384 convolutions with stride [1 1] and padding [1 1].
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        # Conv5: 256 3x3x384 convolutions with stride [1 1] and padding [1 1].
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully Connected Layers.
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc9 = nn.Linear(1024, 4)  # Output predicted parameters (r, g, b, gamma) of shape (batch_size, 4) for color correction.

    def forward(self, x):
        # Save the input image for applying color correction
        input_batch_images = x[:, :3, :, :]

        # Check for NaN in input
        if torch.isnan(x).any():
            logger.error("NaN detected in input to forward pass!")
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Conv4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Conv5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout1(x)
    
        x = self.fc7(x)
        x = F.relu(x)
        x = self.dropout2(x)
    
        x = self.fc8(x)
        x = F.relu(x)
        x = self.dropout3(x)
    
        params = self.fc9(x)

        # Apply constraints to parameters
        r, g, b, gamma = torch.split(params, 1, dim=1)

        # Use exp with clipping for more stable initialization
        r = torch.clamp(F.softplus(r), min=0.1, max=10.0)
        g = torch.clamp(F.softplus(g), min=0.1, max=10.0)
        b = torch.clamp(F.softplus(b), min=0.1, max=10.0)
        gamma = torch.clamp(F.softplus(gamma), min=0.1, max=10.0)
    
        params = torch.cat([r, g, b, gamma], dim=1)

        # Apply color correction to the input batch_images
        corrected_image = self.color_correction(input_batch_images, params)

        return corrected_image

    def color_correction(self, batch_images, params):
        """
        Apply color correction and gamma correction with numerical stability safeguards.
        """
        batch_size, _, H, W = batch_images.shape

        # Extract parameters and add small epsilon to prevent division by zero
        epsilon = 1e-5
        r = params[:, 0] + epsilon  # Ensure r is never exactly zero
        g = params[:, 1] + epsilon
        b = params[:, 2] + epsilon
        gamma = torch.clamp(params[:, 3], min=0.1, max=10.0)  # Restrict gamma to reasonable range

        # Create the diagonal matrix M with clamped values to prevent extreme scaling
        M = torch.zeros(batch_size, 3, 3, device=batch_images.device)
    
        # Calculate 1/r, 1/g, 1/b with clamping to prevent extreme values
        r_inv = torch.clamp(1.0 / r, min=0.01, max=100.0)
        g_inv = torch.clamp(1.0 / g, min=0.01, max=100.0)
        b_inv = torch.clamp(1.0 / b, min=0.01, max=100.0)
    
        M[:, 0, 0] = r_inv
        M[:, 1, 1] = g_inv
        M[:, 2, 2] = b_inv

        # Reshape batch_images for matrix multiplication
        batch_images_reshaped = batch_images.view(batch_size, 3, -1)

        # Apply color correction: I_corrected = M * I
        corrected_batch_images = torch.bmm(M, batch_images_reshaped)

        # Reshape back to (batch_size, 3, H, W)
        corrected_batch_images = corrected_batch_images.view(batch_size, 3, H, W)

        # Clamp values before gamma correction to ensure valid range
        corrected_batch_images = torch.clamp(corrected_batch_images, min=0.0, max=1.0)

        # Apply gamma correction: I_final = I_corrected^gamma
        corrected_batch_images = torch.pow(corrected_batch_images.clamp(min=1e-10), gamma.view(-1, 1, 1, 1))

        # Final safety clamp
        return torch.clamp(corrected_batch_images, min=0.0, max=1.0)


# Configure logger
logger.add("dummy_test.log", format="{time} {level} {message}", level="DEBUG", rotation="5 MB", retention="1 days", compression="zip")


# Create a dummy dataset
def create_dummy_data(num_samples=100, input_size=(4, 227, 227)):
    logger.info(f"Creating dummy dataset with {num_samples} samples and input size {input_size}...")
    X = torch.randn(num_samples, *input_size)  # Random input data (4 channels)
    y = torch.randn(num_samples, 3, 227, 227)  # Random target images (RGB)
    logger.success("Dummy dataset created.")
    return X, y


# Test the model
def test_model():
    logger.info("Starting model test...")
    try:
        # Initialize model
        model = TrueColorNet()
        logger.info("Model initialized successfully.")

        # Dummy data
        X, y = create_dummy_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        logger.info("Loss and optimizer defined.")

        # Training loop
        model.train()
        epochs = 2
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}...")
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
                optimizer.zero_grad()

                # Get the corrected image from the model
                corrected_image = model(inputs)

                # Calculate loss between corrected image and target image
                loss = criterion(corrected_image, targets)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                logger.debug(f"Batch {i + 1}: Loss = {loss.item():.4f}")

            logger.info(f"Epoch {epoch + 1} completed. Average loss: {running_loss / len(dataloader):.4f}")

        logger.success("Model test completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred during model testing: {e}")


# Run the test
if __name__ == "__main__":
    test_model()