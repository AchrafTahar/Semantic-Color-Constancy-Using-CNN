import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F


class TrueColorNet(nn.Module):
    def __init__(self):
        super(TrueColorNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv1: 96 11x11x4 convolutions with stride [4 4] and padding [0 0]
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=96, kernel_size=11, stride=4, padding=0)

        # Conv2: 256 5x5x48 convolutions
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)

        # Conv3: 384 3x3x256 convolutions
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        # Conv4: 384 3x3x384 convolutions
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        # Conv5: 256 3x3x192 convolutions
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc9 = nn.Linear(1024, 4)  # Output 4 parameters for color correction

    def forward(self, x):
        # Save the input image for correction (assume first 3 channels are RGB)
        input_image = x[:, :3, :, :]

        # Conv1
        x = F.relu(self.conv1(x))  # Output shape: (batch_size, 96, H', W')
        x = self.pool(x)  # Output shape: (batch_size, 96, H'', W'')

        # Conv2
        x = F.relu(self.conv2(x))  # Output shape: (batch_size, 256, H'', W'')
        x = self.pool(x)

        # Conv3
        x = F.relu(self.conv3(x))  # Output shape: (batch_size, 384, H'', W'')

        # Conv4
        x = F.relu(self.conv4(x))  # Output shape: (batch_size, 384, H'', W'')

        # Conv5
        x = F.relu(self.conv5(x))  # Output shape: (batch_size, 256, H'', W'')
        x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Fully Connected Layers
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        x = F.relu(self.fc8(x))
        x = self.dropout3(x)
        params = self.fc9(x)  # Output 4 parameters for color correction

        # Apply color correction to the input image
        corrected_image = self.color_correction(input_image, params)

        return corrected_image

    def color_correction(self, image, params):
        """
        Applies color correction and gamma correction to the input image using the predicted four values.

        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).
            params (torch.Tensor): Predicted parameters (r, g, b, γ) of shape (batch_size, 4).

        Returns:
            torch.Tensor: Corrected image tensor of shape (batch_size, 3, H, W).
        """
        batch_size, _, H, W = image.shape

        # Extract parameters
        r, g, b, gamma = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        # Create the diagonal matrix M
        M = torch.zeros(batch_size, 3, 3, device=image.device)
        M[:, 0, 0] = 1.0 / r  # 1/r
        M[:, 1, 1] = 1.0 / g  # 1/g
        M[:, 2, 2] = 1.0 / b  # 1/b

        # Reshape image for matrix multiplication: (batch_size, 3, H*W)
        image_reshaped = image.view(batch_size, 3, -1)

        # Apply color correction: I_corrected = M * I
        corrected_image = torch.bmm(M, image_reshaped)  # (batch_size, 3, H*W)

        # Reshape back to (batch_size, 3, H, W)
        corrected_image = corrected_image.view(batch_size, 3, H, W)

        # Apply gamma correction: I_final = I_corrected^gamma
        corrected_image = torch.pow(corrected_image, gamma.view(-1, 1, 1, 1))

        return corrected_image


# Configure logger
logger.add("model_test.log", format="{time} {level} {message}", level="DEBUG")


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