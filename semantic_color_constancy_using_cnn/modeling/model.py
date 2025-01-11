import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F
# Import the model
class TrueColorNet(nn.Module):
    def __init__(self):
        super(TrueColorNet, self).__init__()

        # Conv1: 96 11x11x4 convolutions with stride [4 4] and padding [0 0]
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv2: 256 5x5x48 convolutions
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=256, kernel_size=5, padding=2)

        # Conv3: 384 3x3x256 convolutions
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        # Conv4: 384 3x3x384 convolutions
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        # Channel reduction: 384 -> 192
        self.reduce_channels = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1)

        # Conv5: 256 3x3x192 convolutions
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc6 = nn.Linear(256 * 26 * 26, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc9 = nn.Linear(1024, 4)  # Output 4 parameters for color correction

    def forward(self, x):
        # Conv1
        x = F.relu(self.conv1(x))  # Output shape: (batch_size, 96, H', W')
        x = self.pool1(x)  # Output shape: (batch_size, 96, H'', W'')

        # Reduce channels from 96 to 48 for Conv2
        x = x[:, :48, :, :]  # Manually slice to 48 channels

        # Conv2
        x = F.relu(self.conv2(x))  # Output shape: (batch_size, 256, H'', W'')

        # Conv3
        x = F.relu(self.conv3(x))  # Output shape: (batch_size, 384, H'', W'')

        # Conv4
        x = F.relu(self.conv4(x))  # Output shape: (batch_size, 384, H'', W'')

        # Channel reduction: 384 -> 192
        x = self.reduce_channels(x)  # Output shape: (batch_size, 192, H'', W'')

        # Conv5
        x = F.relu(self.conv5(x))  # Output shape: (batch_size, 256, H'', W'')

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Fully Connected Layers
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        x = F.relu(self.fc8(x))
        x = self.dropout3(x)
        x = self.fc9(x)  # Output 4 parameters for color correction

        return x

# Configure logger
logger.add("model_test.log", format="{time} {level} {message}", level="DEBUG")

# Create a dummy dataset
def create_dummy_data(num_samples=100, input_size=(4, 224, 224)):
    logger.info(f"Creating dummy dataset with {num_samples} samples and input size {input_size}...")
    X = torch.randn(num_samples, *input_size)  # Random input data
    y = torch.randn(num_samples, 4)  # Random targets (r, g, b, gamma)
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
                outputs = model(inputs)
                loss = criterion(outputs, targets)
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
